"""Check the dataset-overview helpers: stream classification, GIF, frame range.

These exist because a workflow meets datasets it was not written for -- an ADF
detector instead of a HAADF one, an EDS spectrum image alongside the EELS, a
series whose last frames are beam damaged. Each check below is a case that
broke, or would have.

    python -m gatan_insitu.test_notebook
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import hyperspy.api as hs  # noqa: E402

from .notebook import (  # noqa: E402
    classify_signals,
    frame_range,
    image_streams,
    make_navigator,
    quicklook,
    save_frame_gif,
)

_results = []


def check(name, condition, detail=""):
    _results.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return condition


def _signal(data, names, sig_dim, title, eels=False):
    axes = []
    for size, name in zip(data.shape, names):
        axes.append({
            "name": name, "size": size, "scale": 1.0, "offset": 0.0,
            "units": "nm" if name in ("y", "x") else ("eV" if "Energy" in name else "s"),
            "navigate": True,
        })
    for axis in axes[-sig_dim:]:
        axis["navigate"] = False
    maker = hs.signals.Signal1D if sig_dim == 1 else hs.signals.Signal2D
    signal = maker(data, axes=axes)
    if eels:
        signal.set_signal_type("EELS")
    signal.metadata.General.title = title
    return signal


def build_dataset(frames=6, y=12, x=14, channels=64, eds=32, detector="ADF"):
    """A dataset shaped like a real one: a non-HAADF detector, an EDS spectrum
    image in its native layout, derived image series, and dual EELS."""
    rng = np.random.default_rng(0)
    ramp = np.linspace(1.0, 0.6, frames)[:, None, None]
    return {
        f"{detector} Image (SI Survey)":
            _signal(rng.random((64, 64)), ["y", "x"], 2, f"{detector} Survey"),
        "Correlation Coefficient":
            _signal(rng.random((frames, y, x)), ["Time", "y", "x"], 2, "Correlation"),
        "Peak Shift":
            _signal(rng.random((frames, y, x)), ["Time", "y", "x"], 2, "Peak Shift"),
        f"{detector} Image":
            _signal(rng.random((frames, y * 4, x * 4)) * ramp,
                    ["Time", "y", "x"], 2, f"STEM SI_{detector} Image"),
        # nav = (time, E), sig = (y, x): two navigation axes, so data[0] is a
        # stack, not a picture.  This is what broke quicklook.
        "EDS Spectrum Image":
            _signal(rng.random((frames, eds, y, x)),
                    ["Time", "Energy", "y", "x"], 2, "STEM SI_EDS"),
        "EELS HL":
            _signal(rng.random((frames, y, x, channels)),
                    ["Time", "y", "x", "Energy loss"], 1, "EELS HL", eels=True),
        "EELS LL":
            _signal(rng.random((frames, y, x, channels)),
                    ["Time", "y", "x", "Energy loss"], 1, "EELS LL", eels=True),
    }


def main():
    signals = build_dataset()
    spectra, series, stills, other = classify_signals(signals)

    print("stream classification:")
    check("spectra found by signal dimension, not by name",
          set(spectra) == {"EELS HL", "EELS LL"}, str(sorted(spectra)))
    check("image series are 2D signals with ONE navigation axis",
          set(series) == {"Correlation Coefficient", "Peak Shift", "ADF Image"},
          str(sorted(series)))
    check("the survey is a still", set(stills) == {"ADF Image (SI Survey)"},
          str(sorted(stills)))
    check("a 2D-signal spectrum image is NOT mistaken for an image series",
          set(other) == {"EDS Spectrum Image"}, str(sorted(other)))

    order = image_streams(signals)
    check("the detector image comes first, whatever it is called",
          order[0] == "ADF Image", str(order))
    haadf_set = build_dataset(detector="HAADF")
    check("the same holds for HAADF",
          image_streams(haadf_set)[0] == "HAADF Image",
          str(image_streams(haadf_set)[:2]))
    check("derived series are kept, but behind the detector",
          set(order[1:]) == {"Correlation Coefficient", "Peak Shift"}, str(order))

    print("\nquicklook on a dataset it was not written for:")
    try:
        figure = quicklook(signals)
        check("does not raise on a mixed ADF + EDS + dual-EELS dataset", True)
        check("the figure has both rows", len(figure.axes) >= 2,
              f"{len(figure.axes)} axes")
    except Exception as error:
        check("does not raise on a mixed ADF + EDS + dual-EELS dataset", False,
              f"{type(error).__name__}: {error}")
    try:
        quicklook(signals, images=["ADF Image"])
        check("images= chooses the panels", True)
    except Exception as error:
        check("images= chooses the panels", False, repr(error))

    print("\nframe ranges:")
    check("the default is every frame", frame_range(10) == list(range(10)))
    check("stop=0 means to the end", frame_range(10, 3) == [3, 4, 5, 6, 7, 8, 9])
    check("an explicit stop is exclusive", frame_range(10, 2, 5) == [2, 3, 4])
    check("negatives count from the end", frame_range(10, 0, -2) == list(range(8)),
          str(frame_range(10, 0, -2)))
    check("out-of-range values are clipped", frame_range(5, 0, 99) == list(range(5)))
    for bad in ((5, 4, 2), (5, 3, 3)):
        try:
            frame_range(*bad)
            check(f"an empty range {bad[1:]} is refused", False)
        except ValueError:
            check(f"an empty range {bad[1:]} is refused", True)

    print("\nanimated GIF:")
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "frames.png"      # wrong suffix on purpose
        written = save_frame_gif(signals["ADF Image"], target, fps=4)
        check("the suffix is forced to .gif", written.suffix == ".gif", written.name)
        check("the file is written and is a GIF",
              written.exists() and written.read_bytes()[:6] in (b"GIF87a", b"GIF89a"),
              f"{written.stat().st_size} bytes")

        subset = frame_range(signals["ADF Image"].data.shape[0], 1, 4)
        small = save_frame_gif(signals["ADF Image"], Path(folder) / "subset.gif",
                               frames=subset)
        check("a frame subset makes a smaller file",
              small.stat().st_size < written.stat().st_size,
              f"{small.stat().st_size} vs {written.stat().st_size} bytes")

        plain = save_frame_gif(signals["ADF Image"], Path(folder) / "plain.gif",
                               label=False)
        check("labelling changes the pixels (the number really is burnt in)",
              plain.read_bytes() != written.read_bytes())

        try:
            save_frame_gif(signals["EDS Spectrum Image"], Path(folder) / "bad.gif")
            check("a non-image-series is refused", False)
        except ValueError as error:
            check("a non-image-series is refused", True, str(error)[:52])
        try:
            save_frame_gif(signals["ADF Image"], Path(folder) / "none.gif", frames=[])
            check("an empty frame list is refused", False)
        except ValueError:
            check("an empty frame list is refused", True)

    print("\nnavigator from whatever detector is there:")
    nav = make_navigator(signals["EELS HL"], image=signals["ADF Image"])
    check("image= builds a navigator on the SI grid",
          nav.data.shape == signals["EELS HL"].data.shape[:3], str(nav.data.shape))
    old = make_navigator(signals["EELS HL"], haadf=signals["ADF Image"])
    check("haadf= still works as an alias",
          np.array_equal(np.asarray(old.data), np.asarray(nav.data)))
    check("the navigator is titled after the stream it came from",
          "ADF" in nav.metadata.General.title, nav.metadata.General.title)

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
