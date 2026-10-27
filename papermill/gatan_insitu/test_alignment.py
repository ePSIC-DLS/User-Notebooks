"""Check frame-drift alignment against a series drifted on purpose.

A textured specimen is imaged through a moving window: the HAADF at the fine
sampling, the SI at 4x coarser, exactly as recorded.  The drift is planted, so
the right answer is known -- and it is chosen to break the convenient cases:
sub-pixel, not a multiple of the 4x grid, changing sign, with the specimen
itself evolving through the series.  Each SI pixel's "spectrum" carries the
true specimen coordinate it saw, so registration is checked exactly, not by
eye: after alignment every pixel must see the same place in every frame.

    python -m gatan_insitu.test_alignment
"""

from __future__ import annotations

import sys

import hyperspy.api as hs
import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

from .alignment import align_frames, frame_drift

_results = []
RATIO = 4


def check(name, condition, detail=""):
    _results.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return condition


def build(n_frames=8, si_shape=(30, 34), seed=3, evolve=0.3, noise=0.02):
    """HAADF + SI series of one specimen seen through a drifting window."""
    rng = np.random.default_rng(seed)
    fine = (si_shape[0] * RATIO, si_shape[1] * RATIO)
    margin = 40
    scene = gaussian_filter(rng.normal(size=(fine[0] + 2 * margin, fine[1] + 2 * margin)), 3)
    growth = gaussian_filter(rng.normal(size=scene.shape), 8)  # the specimen changes

    # Window origin in scene pixels: sub-pixel, sign-changing, not multiples of 4.
    t = np.arange(n_frames)
    origin = np.stack([7.3 * np.sin(t / 2.5) + 0.9 * t, -1.7 * t + 3.1 * np.cos(t)], axis=1)
    truth = -(origin - origin[0])  # feature displacement in the image = -window motion

    haadf, si = [], []
    ys, xs = np.mgrid[0:fine[0], 0:fine[1]].astype(float)
    for k in range(n_frames):
        specimen = scene + evolve * (k / max(n_frames - 1, 1)) * growth
        view = nd_shift(specimen, -origin[k], order=3, mode="nearest")[margin:margin + fine[0],
                                                                       margin:margin + fine[1]]
        view = view + noise * rng.normal(size=view.shape)
        haadf.append(view)
        # Specimen coordinate (scene frame) seen at each fine pixel, binned onto the SI.
        coord_y = (ys + origin[k, 0]).reshape(si_shape[0], RATIO, si_shape[1], RATIO).mean((1, 3))
        coord_x = (xs + origin[k, 1]).reshape(si_shape[0], RATIO, si_shape[1], RATIO).mean((1, 3))
        image = view.reshape(si_shape[0], RATIO, si_shape[1], RATIO).sum((1, 3))
        si.append(np.stack([coord_y, coord_x, image], axis=-1))

    nav = lambda n, name, size: {"name": name, "size": size, "scale": n, "offset": 0.0,
                                 "units": "nm", "navigate": True}
    time = {"name": "Time", "size": n_frames, "scale": 18.0, "offset": 0.0, "units": "s",
            "navigate": True}
    h = hs.signals.Signal2D(np.array(haadf, dtype=np.float32), axes=[
        time,
        {**nav(0.03, "y", fine[0]), "navigate": False},
        {**nav(0.03, "x", fine[1]), "navigate": False},
    ])
    s = hs.signals.Signal1D(np.array(si, dtype=np.float64), axes=[
        time, nav(0.12, "y", si_shape[0]), nav(0.12, "x", si_shape[1]),
        {"name": "Energy loss", "size": 3, "scale": 1.0, "offset": 0.0, "units": "eV",
         "navigate": False},
    ])
    h.metadata.General.title = "HAADF"
    return h, s, truth


def spread(si_data):
    """Largest frame-to-frame scatter of the specimen coordinate any pixel sees (fine px)."""
    coords = si_data[..., :2]
    return float(np.max(coords.max(axis=0) - coords.min(axis=0)))


def main():
    print("frame drift alignment")
    haadf, si, truth = build()

    measured = frame_drift(haadf)
    error = np.abs(measured - truth).max()
    check("drift is measured to sub-pixel accuracy, both signs, specimen evolving",
          error < 0.5, f"max error {error:.2f} fine px over a {np.ptp(truth, axis=0)} px drift")
    first = frame_drift(haadf, reference="first")
    check("'first' reference agrees on a mildly evolving specimen",
          np.abs(first - truth).max() < 0.5, f"max error {np.abs(first - truth).max():.2f} px")

    before = spread(si.data)
    haadf_a, si_a, report = align_frames(haadf.as_lazy(), si.as_lazy())
    check("signals stay lazy", haadf_a._lazy and si_a._lazy)
    si_a.compute()
    haadf_a.compute()
    after = spread(si_a.data)
    # Integer SI shifts leave each frame within half an SI pixel, so two frames can
    # differ by up to one SI pixel: the scatter is the spread of the residuals.
    expected = float(np.ptp(report["residual_px"], axis=0).max())
    check("every SI pixel sees the same specimen place, to the rounding residual",
          after <= expected + 0.5 and after <= RATIO + 0.5,
          f"coordinate scatter {before:.1f} -> {after:.2f} fine px "
          f"(residual spread {expected:.2f}, one SI pixel = {RATIO})")
    check("reported residual matches the real one",
          np.abs(report["residual_px"]).max() <= RATIO / 2 + 0.5,
          f"max |residual| {np.abs(report['residual_px']).max():.2f} fine px")

    rebinned = haadf_a.data.reshape(haadf_a.data.shape[0], si_a.data.shape[1], RATIO,
                                    si_a.data.shape[2], RATIO).sum((2, 4))
    check("aligned HAADF still rebins exactly onto the aligned SI",
          np.allclose(rebinned, si_a.data[..., 2], atol=1e-3 * np.abs(rebinned).max()),
          f"HAADF {haadf_a.data.shape[1:]} vs SI {si_a.data.shape[1:3]}")

    lo, hi = report["applied_shifts_coarse_px"].min(0), report["applied_shifts_coarse_px"].max(0)
    check("field is cropped by exactly the drift range",
          si_a.data.shape[1:3] == tuple(np.array(si.data.shape[1:3]) - (hi - lo)),
          f"{si.data.shape[1:3]} -> {si_a.data.shape[1:3]}, kept {100 * report['field_kept']:.0f}%")
    check("spectra are not interpolated",
          np.isin(si_a.data[..., 2], si.data[..., 2]).all())
    offsets = si_a.axes_manager["y"].offset, si_a.axes_manager["x"].offset
    check("spatial calibration follows the crop",
          lo[0] < 0 and np.allclose(offsets, -lo * 0.12),
          f"(y, x) offset {offsets[0]:.2f}, {offsets[1]:.2f} nm for shifts down to {lo}")
    check("shifts are recorded in the metadata",
          si_a.metadata.has_item("Processing.frame_alignment.shifts_px"))

    try:
        bad = si.isig[:].inav[:, :-1]  # 29 rows: not a quarter of 120
        align_frames(haadf, bad)
        check("a non-integer grid ratio is refused", False)
    except ValueError as error:
        check("a non-integer grid ratio is refused", True, str(error)[:60] + "...")

    one = haadf.inav[0:1]
    one_si = si.inav[:, :, 0:1]
    _, s1, r1 = align_frames(one, one_si)
    check("a single frame passes through unchanged",
          s1.data.shape == one_si.data.shape and r1["field_kept"] == 1.0)

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
