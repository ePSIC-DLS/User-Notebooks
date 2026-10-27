"""Check segmentation and masked extraction against data with a known answer.

Builds a synthetic series whose regions, areas and per-region spectra are known
exactly, so the masks and the extracted spectra can be compared with the truth
rather than merely eyeballed. Also checks the moving-region case that motivates
per-frame masking: a region that drifts and grows across the series.

    python -m gatan_insitu.test_segmentation
"""

from __future__ import annotations

import sys

import dask.array as da
import hyperspy.api as hs
import numpy as np

from .segmentation import (
    _overlay_coords,
    compare_regions,
    masked_spectra,
    rebin_masks,
    segment_frames,
)

_results = []


def check(name, condition, detail=""):
    _results.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return condition


def build(n_frames=8, size=48, n_energy=64, seed=0):
    """A disc that drifts and grows on a plateau, each region with its own spectrum."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    truth = np.zeros((n_frames, size, size), dtype=np.uint8)
    for frame in range(n_frames):
        cy = 16 + frame                        # drifts
        cx = 16 + frame
        plateau = (np.abs(yy - cy) < 14) & (np.abs(xx - cx) < 14)
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 < (4 + 0.5 * frame) ** 2  # grows
        truth[frame][plateau] = 1
        truth[frame][disc] = 2

    # Distinct, noiseless spectra per class so the extraction is exactly checkable.
    energy = np.arange(n_energy, dtype=np.float64)
    spectra = np.stack([
        np.full(n_energy, 1.0),
        2.0 + np.exp(-((energy - 20) ** 2) / 8.0) * 5.0,
        3.0 + np.exp(-((energy - 45) ** 2) / 8.0) * 9.0,
    ])

    cube = spectra[truth]                                   # (F, Y, X, E)
    source = np.array([[0.0, 5.0, 12.0][k] for k in truth.ravel()]).reshape(truth.shape)
    source = source + rng.normal(0, 0.25, source.shape)     # a little noise
    return truth, spectra, cube, source


def main():
    truth, spectra, cube, source = build()
    n_frames = truth.shape[0]
    print(f"synthetic series: {cube.shape}, regions move and grow")

    masks = segment_frames(
        source, n_classes=3, labels=["background", "plateau", "disc"],
        smooth=0.8, min_size=4,
    )

    check("mask shape", masks.masks.shape == (n_frames, 3) + truth.shape[1:],
          str(masks.masks.shape))
    check("masks are mutually exclusive and cover the frame",
          bool(np.all(masks.masks.sum(axis=1) == 1)))

    agreement = (masks.label_image() == truth).mean()
    check("masks recover the truth", agreement > 0.98, f"{100 * agreement:.2f}% of pixels")

    # The point of per-frame masking: a single mask would not track the motion.
    static = np.broadcast_to(masks.masks[0], masks.masks.shape)
    static_agreement = (static.argmax(axis=1) == truth).mean()
    check("per-frame masks beat a single fixed mask", agreement > static_agreement + 0.05,
          f"per-frame {100 * agreement:.1f}% vs frame-0 mask {100 * static_agreement:.1f}%")

    areas = masks.areas
    true_areas = np.stack([(truth == k).sum(axis=(1, 2)) for k in range(3)], axis=1)
    check("areas within 2% of truth",
          np.all(np.abs(areas - true_areas) <= 0.02 * true_areas + 8),
          f"disc {areas[:, 2].tolist()} vs {true_areas[:, 2].tolist()}")
    check("the disc is seen to grow", areas[-1, 2] > 1.5 * areas[0, 2],
          f"{areas[0, 2]} -> {areas[-1, 2]} px")

    # Masked extraction, against a lazy signal chunked the way the loader does.
    signal = hs.signals.Signal1D(da.from_array(cube, chunks=(1, -1, -1, 16))).as_lazy()
    signal.axes_manager._axes[0].name = "Time"
    signal.axes_manager._axes[1].name = "y"
    signal.axes_manager._axes[2].name = "x"
    signal.axes_manager.signal_axes[0].name = "Energy loss"

    got = masked_spectra(signal, masks, mode="mean")
    check("one series per label", sorted(got) == ["background", "disc", "plateau"],
          str(sorted(got)))
    check("not lazy", not getattr(got["disc"], "_lazy", False),
          type(got["disc"]).__name__)

    # Where a mask is exactly right, the mean spectrum must be the pure one.
    for klass, label in enumerate(["background", "plateau", "disc"]):
        pure = np.array([
            cube[f][masks.masks[f, klass]].mean(axis=0) for f in range(n_frames)
        ])
        error = np.abs(got[label].data - pure).max()
        check(f"'{label}' mean matches a direct masked mean",
              error < 1e-4, f"max |diff| = {error:.2e}")

    disc_error = np.abs(got["disc"].data - spectra[2]).max()
    check("disc spectrum equals the pure disc spectrum",
          disc_error < 0.3, f"max |diff| = {disc_error:.3f} (edge pixels blend)")

    # sum mode must equal mean x pixel count
    totals = masked_spectra(signal, masks, mode="sum")
    product = got["plateau"].data * masks.areas[:, 1][:, None]
    check("sum == mean x pixels",
          np.allclose(totals["plateau"].data, product, rtol=1e-4),
          f"max |diff| = {np.abs(totals['plateau'].data - product).max():.2e}")

    # compare_regions must call these genuinely different (not just scaled)
    report = compare_regions(got, "plateau", "disc", pre_window=(0, 10),
                             edges={"peak": ((35, 42), (43, 50))}, plot=False)
    check("compare_regions sees a real difference", report["flatness"] > 0.2,
          f"residual peak = {100 * report['flatness']:.0f}% of signal")

    # ... and must call a pure thickness difference flat.
    scaled = {k: v.deepcopy() for k, v in got.items()}
    scaled["disc"].data = got["plateau"].data * 2.5
    flat = compare_regions(scaled, "plateau", "disc", pre_window=(0, 10), plot=False)
    check("compare_regions calls a pure scaling flat", flat["flatness"] < 1e-6,
          f"residual peak = {flat['flatness']:.2e}, scale {flat['scale']:.2f}")

    # Mismatched grids must be refused, not silently broadcast.
    try:
        masked_spectra(signal, segment_frames(source[:, :24, :24], n_classes=3))
        check("mismatched mask grid refused", False)
    except ValueError as error:
        check("mismatched mask grid refused", True, str(error)[:60])

    print("\nsegmenting at HAADF resolution, then reducing onto the SI grid:")
    fine_source = np.repeat(np.repeat(source, 4, axis=1), 4, axis=2)  # 4x finer
    fine = segment_frames(fine_source, n_classes=3,
                          labels=["background", "plateau", "disc"],
                          smooth=3.2, min_size=64)
    check("fine masks are at the fine resolution",
          fine.masks.shape[2:] == fine_source.shape[1:], str(fine.masks.shape))

    coarse = rebin_masks(fine, target_shape=truth.shape[1:], mode="fraction")
    check("reduced onto the coarse grid",
          coarse.masks.shape == (n_frames, 3) + truth.shape[1:], str(coarse.masks.shape))
    check("fractional weights", coarse.is_fractional, str(coarse.masks.dtype))
    check("weights sum to 1 per pixel", np.allclose(coarse.masks.sum(axis=1), 1.0))
    check("area fraction is preserved by the reduction",
          np.allclose(coarse.area_fraction, fine.area_fraction, atol=1e-6),
          f"max diff {np.abs(coarse.area_fraction - fine.area_fraction).max():.2e}")
    check("some pixels really do straddle a boundary",
          ((coarse.masks > 0.01).sum(axis=1) > 1).any(),
          f"{100 * np.mean((coarse.masks > 0.01).sum(axis=1) > 1):.1f}% of pixels")

    majority = rebin_masks(fine, target_shape=truth.shape[1:], mode="majority")
    check("majority mode gives boolean masks",
          not majority.is_fractional and bool(np.all(majority.masks.sum(axis=1) == 1)),
          str(majority.masks.dtype))

    # Fractional weights must produce an exactly weighted average.
    weighted = masked_spectra(signal, coarse, mode="mean")
    klass = 2
    manual = []
    for frame in range(n_frames):
        w = coarse.masks[frame, klass]
        manual.append((cube[frame] * w[..., None]).sum(axis=(0, 1)) / w.sum())
    error = np.abs(weighted["disc"].data - np.array(manual)).max()
    check("fractional masked mean is exactly weighted", error < 1e-4,
          f"max |diff| = {error:.2e}")

    try:
        rebin_masks(fine, target_shape=(37, 37))
        check("non-integer reduction refused", False)
    except ValueError as error:
        check("non-integer reduction refused", True, str(error)[:60])
    try:
        rebin_masks(fine, factor=4, target_shape=(12, 12))
        check("factor and target_shape together refused", False)
    except ValueError as error:
        check("factor and target_shape together refused", True, str(error)[:50])

    print("\noverlaying masks on an image of a different resolution:")
    # Coarse masks drawn over the finer image they came from must land on the
    # same features, not be squashed into the top-left corner by the binning.
    xs, ys = _overlay_coords((111, 118), (444, 472))
    check("overlay spans the whole image, not 1/factor of it",
          abs(xs[-1] - 469.5) < 1e-9 and abs(ys[-1] - 441.5) < 1e-9,
          f"x to {xs[-1]:.1f} of 471.5, y to {ys[-1]:.1f} of 443.5")
    check("overlay block centres are a half-block in",
          abs(xs[0] - 1.5) < 1e-9 and abs(ys[0] - 1.5) < 1e-9,
          f"first centre ({xs[0]:.1f}, {ys[0]:.1f}) for 4x blocks")
    check("overlay spacing equals the binning factor",
          np.allclose(np.diff(xs), 4.0) and np.allclose(np.diff(ys), 4.0))
    same_x, same_y = _overlay_coords((111, 118), (111, 118))
    check("overlay is the identity when the grids match",
          np.allclose(same_x, np.arange(118)) and np.allclose(same_y, np.arange(111)))

    # The real thing, through plot(): a coarse mask over its fine source.
    import matplotlib
    matplotlib.use("Agg")
    centre_fine = np.array(
        [np.argwhere(fine.masks[f, 2]).mean(axis=0) for f in range(n_frames)]
    )
    coarse_axes = _overlay_coords(coarse.masks.shape[2:], fine_source.shape[1:])
    centre_coarse = []
    for f in range(n_frames):
        w = coarse.masks[f, 2]
        yy = (w.sum(axis=1) * coarse_axes[1]).sum() / w.sum()
        xx = (w.sum(axis=0) * coarse_axes[0]).sum() / w.sum()
        centre_coarse.append([yy, xx])
    offset = np.abs(np.array(centre_coarse) - centre_fine).max()
    check("mapped coarse masks sit on the same features as the fine ones",
          offset < 1.0, f"max centroid offset {offset:.2f} fine pixels")
    try:
        coarse.plot(fine_source, frames=(0, -1))
        check("plot() accepts a source at a different resolution", True)
    except Exception as error:
        check("plot() accepts a source at a different resolution", False, repr(error))
    try:
        coarse.plot(fine_source[:3])
        check("frame-count mismatch refused", False)
    except ValueError as error:
        check("frame-count mismatch refused", True, str(error)[:50])

    print("\neroding masks away from their boundaries:")
    from .segmentation import erode_masks

    shrunk = erode_masks(masks, iterations=2)
    check("eroding shrinks the non-background classes",
          np.all(shrunk.areas[:, 1:] < masks.areas[:, 1:]),
          f"disc {masks.areas[0, 2]} -> {shrunk.areas[0, 2]} px")
    check("the classes still tile the frame",
          bool(np.all(shrunk.masks.sum(axis=1) == 1)))
    check("eroded masks stay inside the originals",
          bool(np.all(shrunk.masks[:, 1:] <= masks.masks[:, 1:])))
    check("what was eroded away became background",
          shrunk.areas[0, 0] > masks.areas[0, 0],
          f"{masks.areas[0, 0]} -> {shrunk.areas[0, 0]} px")
    try:
        erode_masks(coarse)
        check("eroding fractional masks is refused", False)
    except ValueError as error:
        check("eroding fractional masks is refused", True, str(error)[:48])

    # The point of the control: a signal that is really spillover from the
    # neighbour must fall away as the mask shrinks, while an intrinsic one
    # must not.  Build both and check the test can tell them apart.
    spill = np.zeros_like(cube)
    intrinsic = np.zeros_like(cube)
    for frame in range(n_frames):
        disc = masks.masks[frame, 2]
        edge = disc & ~shrunk.masks[frame, 2]
        spill[frame][edge] = spectra[1]        # only at the boundary
        intrinsic[frame][disc] = spectra[2]    # everywhere in the disc

    def level(volume, stack_masks):
        sig = hs.signals.Signal1D(da.from_array(volume, chunks=(1, -1, -1, 16))).as_lazy()
        for axis, name in zip(sig.axes_manager._axes, ["Time", "y", "x", "Energy loss"]):
            axis.name = name
        return np.nanmean(masked_spectra(sig, stack_masks, mode="mean")["disc"].data)

    spill_before, spill_after = level(spill, masks), level(spill, shrunk)
    core_before, core_after = level(intrinsic, masks), level(intrinsic, shrunk)
    check("a boundary-only signal falls away when eroded",
          spill_after < 0.5 * spill_before,
          f"{spill_before:.3f} -> {spill_after:.3f}")
    check("an intrinsic signal survives erosion",
          core_after > 0.9 * core_before,
          f"{core_before:.3f} -> {core_after:.3f}")

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
