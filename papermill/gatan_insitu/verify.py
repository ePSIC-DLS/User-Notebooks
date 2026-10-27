"""Self-check: does the lazily loaded raw stream agree with the .dm4?

A multiframe .dm4 stores one frame of the series -- the last one -- so the
loader can be validated end to end without any external reference: load the
.dm4 through rosettasciio, load the .raw through this package, and compare the
final frame element by element.  Also checks the calibrations, that nothing was
computed eagerly, and that the two read backends agree.

    python -m gatan_insitu.verify "<folder>"
"""

from __future__ import annotations

import sys
from pathlib import Path

import dask.array as da
import hyperspy.api as hs
import numpy as np

from .dm_tags import read_raw_format_info
from .loader import load_insitu_signal
from .raw_store import RawFrameStore

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    _results.append(condition)
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    return condition


def verify_file(dm4_path):
    dm4_path = Path(dm4_path)
    print(f"\n{dm4_path.name}")
    info = read_raw_format_info(dm4_path)
    if info is None:
        print("  (single-frame .dm4, nothing to verify)")
        return
    print(f"  {info.summary()}")

    reference = hs.load(dm4_path, lazy=True)
    reference = reference[-1] if isinstance(reference, list) else reference
    ref_data = np.asarray(reference.data)

    signal = load_insitu_signal(dm4_path, layout="native")
    check("stayed lazy", isinstance(signal.data, da.Array))
    # A dask array can still be backed by an in-RAM copy of the whole file
    # (dask.array.from_array copies any array-like exposing .copy()), so check
    # the graph holds only path references, not bulk data.
    embedded = max(
        (
            value.nbytes
            for value in signal.data.dask.values()
            if isinstance(value, np.ndarray)
        ),
        default=0,
    )
    check(
        "no bulk data embedded in the dask graph",
        embedded < 16 * 1024**2,
        f"largest embedded array {embedded / 1024**2:.1f} MB",
    )
    chunk_mb = int(np.prod(signal.data.chunksize)) * signal.data.dtype.itemsize / 1024**2
    check(
        "chunks are a sane size",
        1 <= chunk_mb <= 256,
        f"chunksize {signal.data.chunksize} = {chunk_mb:.1f} MB, "
        f"{signal.data.npartitions} partitions",
    )

    # The .dm4 holds the last frame of the series.
    last = np.asarray(signal.data[-1].compute())
    if signal.data.ndim == 4:
        # native layout is (time, E, y, x); the .dm4 frame is (y, x, E)
        last = last.transpose(1, 2, 0)
    check(
        "last raw frame is identical to the frame in the .dm4",
        last.shape == ref_data.shape and np.array_equal(last, ref_data),
        f"raw {last.shape} vs dm4 {ref_data.shape}",
    )

    # Calibrations, against what rosettasciio derived from the same file.
    spectrum = load_insitu_signal(dm4_path, layout="spectrum")
    ref_axes = [a for a in reference.axes_manager._axes]
    sig_axes = [a for a in spectrum.axes_manager._axes][1:]  # skip the time axis
    for ref_axis, axis in zip(ref_axes, sig_axes):
        check(
            f"axis '{axis.name}' calibration",
            axis.size == ref_axis.size
            and np.isclose(axis.scale, ref_axis.scale, rtol=1e-9)
            and np.isclose(axis.offset, ref_axis.offset, atol=1e-6)
            and axis.units == ref_axis.units,
            f"{axis.size} @ {axis.scale:.6g} + {axis.offset:.6g} {axis.units} "
            f"(dm4: {ref_axis.size} @ {ref_axis.scale:.6g} + {ref_axis.offset:.6g} "
            f"{ref_axis.units})",
        )

    # The spectrum layout must be a lazy transpose of the native one.
    check(
        "spectrum layout matches native layout",
        np.array_equal(
            np.asarray(spectrum.data[-1].compute()),
            last if signal.data.ndim == 4 else np.asarray(signal.data[-1].compute()),
        ),
    )

    # Both read backends must return the same bytes.
    few = load_insitu_signal(dm4_path, layout="native", frames=2, backend="fromfile")
    mm = load_insitu_signal(dm4_path, layout="native", frames=2, backend="memmap")
    check(
        "'fromfile' and 'memmap' backends agree",
        np.array_equal(np.asarray(few.data.compute()), np.asarray(mm.data.compute())),
    )

    # Frame bookkeeping.
    store = RawFrameStore(info.raw_paths, info.frame_shape, info.dtype)
    total = sum(p.stat().st_size for p in info.raw_paths)
    check(
        "frame count follows the raw file size(s)",
        store.n_frames == total // info.frame_bytes,
        f"{store.n_frames} frames, {total} B / {info.frame_bytes} B",
    )
    check(
        "frame count matches the tagged '# Frames'",
        store.n_frames == info.n_frames_declared,
        f"{store.n_frames} vs {info.n_frames_declared}",
    )

    times = spectrum.metadata.get_item("InSitu.frame_times_s", None)
    if times is not None:
        gaps = np.diff(np.asarray(times))
        check(
            "per-frame timestamps are regular",
            np.all(gaps > 0) and gaps.std() / gaps.mean() < 0.01,
            f"dt = {gaps.mean():.4f} +/- {gaps.std():.4f} s over {len(times)} frames",
        )


def main(folder):
    folder = Path(folder)
    for dm4 in sorted(folder.glob("*.dm4")):
        verify_file(dm4)
    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
