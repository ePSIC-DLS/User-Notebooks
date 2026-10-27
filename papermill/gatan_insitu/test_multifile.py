"""Check the split-raw path, which a single-file dataset never exercises.

Takes a real multiframe acquisition and rewrites its ``.raw`` as several files
under each supported naming convention, then checks that loading the split copy
is byte-identical to loading the original.

The important case is **mid-frame splits**: GMS cuts a long recording at a
fixed byte size (10 GiB in the records seen so far) with no regard for frame
boundaries, so a frame routinely straddles two files. Treating each file as a
whole number of frames loses the straddling frame *and* misaligns every frame
after it, which is silent corruption rather than an error -- so that is the
case this test leads with.

    python -m gatan_insitu.test_multifile "<dm4 path>" [work dir]
"""

from __future__ import annotations

import gc
import shutil
import sys
from pathlib import Path

import numpy as np

from .dm_tags import find_raw_files, read_raw_format_info
from .loader import load_insitu_signal
from .raw_store import LazyRawArray

SPELLINGS = {
    "dot_underscore": lambda stem, i: f"{stem}.raw" if i == 0 else f"{stem}.raw_{i}",
    "dot_dot": lambda stem, i: f"{stem}.raw" if i == 0 else f"{stem}.raw.{i}",
    "stem_number": lambda stem, i: f"{stem}.raw" if i == 0 else f"{stem}_{i}.raw",
}


def split_dataset(dm4_path, out_dir, spelling, byte_edges):
    """Copy ``dm4_path`` into ``out_dir``, cutting its raw stream at ``byte_edges``."""
    dm4_path = Path(dm4_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.iterdir():
        existing.unlink()

    info = read_raw_format_info(dm4_path)
    source = info.raw_paths[0]
    total = source.stat().st_size
    edges = [0] + list(byte_edges) + [total]
    counts = [b - a for a, b in zip(edges, edges[1:])]

    shutil.copy2(dm4_path, out_dir / dm4_path.name)
    namer = SPELLINGS[spelling]
    with open(source, "rb") as src:
        for i, count in enumerate(counts):
            with open(out_dir / namer(dm4_path.stem, i), "wb") as dst:
                remaining = count
                while remaining:
                    block = src.read(min(remaining, 64 << 20))
                    if not block:
                        raise IOError("raw file ended early")
                    dst.write(block)
                    remaining -= len(block)
    return out_dir / dm4_path.name, counts


def main(dm4_path, work_dir=None):
    dm4_path = Path(dm4_path)
    work_dir = Path(work_dir or Path.cwd() / "_split_test")

    info = read_raw_format_info(dm4_path)
    frame_bytes = info.frame_bytes
    total = info.raw_paths[0].stat().st_size
    n_ref = total // frame_bytes

    reference = load_insitu_signal(dm4_path, layout="native")
    full = np.asarray(reference.data.compute())
    print(
        f"reference: {dm4_path.name}, {n_ref} frames of {frame_bytes} B, "
        f"shape {full.shape}"
    )

    results = []

    def check(label, split_dm4, expect_frames, counts):
        found = [p.name for p in find_raw_files(split_dm4)]
        signal = load_insitu_signal(split_dm4, layout="native")
        got = np.asarray(signal.data.compute())
        # Release the memmaps, or Windows will not let the files be replaced.
        for value in signal.data.dask.values():
            if isinstance(value, LazyRawArray):
                value.stream.close()
        ok = (
            len(found) == len(counts)
            and got.shape[0] == expect_frames
            and np.array_equal(got, full[:expect_frames])
        )
        results.append(ok)
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {label}: {counts} B -> "
            f"{len(found)} files, {got.shape[0]} frames"
            + ("" if ok else f"  (expected {expect_frames}, data equal: "
                             f"{np.array_equal(got, full[:expect_frames])})")
        )
        del signal, got
        gc.collect()

    # 1. Mid-frame splits -- the real GMS behaviour, and the regression that
    #    silently misaligned every frame after the boundary.
    print("\nmid-frame splits (frames straddle the boundary):")
    for spelling in SPELLINGS:
        edge = (n_ref // 2) * frame_bytes + frame_bytes // 3      # inside a frame
        split_dm4, counts = split_dataset(
            dm4_path, work_dir / f"mid_{spelling}", spelling, [edge]
        )
        check(spelling, split_dm4, n_ref, counts)

    # Several straddling boundaries at once, unevenly spaced.
    edges = [frame_bytes * 3 + 17, frame_bytes * 11 + frame_bytes // 2, total - 12345]
    split_dm4, counts = split_dataset(
        dm4_path, work_dir / "mid_many", "dot_underscore", edges
    )
    check("four files, three straddles", split_dm4, n_ref, counts)

    # A boundary one byte into a frame, and one byte before its end.
    for tag, (label, edge) in enumerate(
        (("1 B into a frame", frame_bytes * 5 + 1),
         ("1 B before a frame end", frame_bytes * 6 - 1))
    ):
        split_dm4, counts = split_dataset(
            dm4_path, work_dir / f"mid_edge{tag}", "dot_underscore", [edge]
        )
        check(label, split_dm4, n_ref, counts)

    # 2. Frame-aligned splits must keep working.
    print("\nframe-aligned splits:")
    edges = [frame_bytes * 15, frame_bytes * 38]
    split_dm4, counts = split_dataset(
        dm4_path, work_dir / "aligned", "dot_underscore", edges
    )
    check("three files on frame boundaries", split_dm4, n_ref, counts)

    # 3. A truncated recording: the partial final frame is dropped, and the
    #    frames before it must still be intact.
    print("\ntruncation:")
    split_dm4, counts = split_dataset(
        dm4_path, work_dir / "truncated", "dot_underscore",
        [(n_ref // 2) * frame_bytes + 999],
    )
    last = sorted((work_dir / "truncated").glob("*.raw_*"))[-1]
    with open(last, "r+b") as handle:
        handle.truncate(last.stat().st_size - 1024)
    check("partial final frame dropped", split_dm4, n_ref - 1, counts)

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
