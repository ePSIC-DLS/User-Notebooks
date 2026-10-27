"""Assemble one or more headerless ``.raw`` frame files into a dask array.

**The files are one continuous byte stream, not a sequence of whole frames.**
GMS splits a long recording at a fixed byte size -- 10 GiB exactly, in the
records seen so far -- with no regard for where frames begin, so a frame
routinely straddles two files::

    STEM SI_EELS HL SI.raw     10737418240 B = 66 frames + 62525440 B
    STEM SI_EELS HL SI.raw_1    7215810560 B = 44 frames + 99215360 B
                                               ^^^^^^^^   ^^^^^^^^
                                       these two add up to exactly one frame

Counting frames per file would both lose the straddling frame and misalign
every frame after it. So the files are concatenated logically and the frame
grid is laid over the result: ``n_frames = total_bytes // frame_bytes``.

Chunking follows the file layout rather than the logical axis order.  Each
frame of a spectrum image is stored as ``(E, y, x)``, i.e. a stack of
energy-slice images, so a chunk spanning all of ``y`` and ``x`` and a block of
``E`` is one contiguous run of bytes.  The transpose to the ``(..., E)`` order
HyperSpy wants happens afterwards in the dask graph and costs nothing at read
time.
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path

import dask.array as da
import numpy as np

from .dm_tags import stable_hash

_logger = logging.getLogger(__name__)

#: Target chunk size.  16 MB measured fastest on this data for both access
#: patterns: a full 9 GB reduction ran at 1.85 GB/s against 1.0 GB/s at 64 MB,
#: and a narrow energy window reads 18% of the file instead of 67%, because the
#: energy axis is cut into 11 chunks rather than 3.
DEFAULT_CHUNK_MB = 16


def _pick_chunks(frame_shape, itemsize, chunk_mb, max_frames):
    """Choose ``(frames_per_chunk, within_frame_chunks)`` for a target size.

    Small frames are grouped together; frames larger than the target are split
    along their slowest axis only.  Either way the resulting block is a single
    contiguous run of bytes, which :class:`LazyRawArray` relies on.
    """
    target = int(chunk_mb * 1024 * 1024)
    frame_bytes = int(np.prod(frame_shape)) * itemsize

    if frame_bytes <= target:
        per_chunk = max(1, target // frame_bytes)
        return min(int(per_chunk), max(1, max_frames)), tuple(frame_shape)

    slab_bytes = itemsize * int(np.prod(frame_shape[1:])) or itemsize
    slabs = max(1, target // slab_bytes)
    n_chunks = max(1, math.ceil(frame_shape[0] / slabs))
    size = math.ceil(frame_shape[0] / n_chunks)
    return 1, (size,) + tuple(frame_shape[1:])


class RawByteStream:
    """The ``.raw`` files of one acquisition, addressed as one byte stream.

    Reads are served by whole-file ``uint8`` memmaps (``backend="memmap"``) or
    by explicit seeks (``backend="fromfile"``, steadier over a network share).
    Working in bytes rather than elements keeps this correct even when a file
    boundary falls inside a single value.
    """

    def __init__(self, paths, backend="memmap"):
        if backend not in ("memmap", "fromfile"):
            raise ValueError(f"backend must be 'memmap' or 'fromfile', got {backend!r}")
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise FileNotFoundError("no .raw file given")
        self.sizes = [p.stat().st_size for p in self.paths]
        self.starts = np.cumsum([0] + self.sizes)
        self.backend = backend
        self._maps = {}

    @property
    def nbytes(self):
        return int(self.starts[-1])

    def _map(self, index):
        memmap = self._maps.get(index)
        if memmap is None:
            memmap = np.memmap(self.paths[index], dtype=np.uint8, mode="r")
            self._maps[index] = memmap
        return memmap

    def read(self, start, count):
        """Return ``count`` bytes from offset ``start`` in the joined stream."""
        if start < 0 or start + count > self.nbytes:
            raise EOFError(
                f"read of {count} B at {start} runs past the end of the "
                f"{self.nbytes} B stream"
            )
        out = np.empty(count, dtype=np.uint8)
        written = 0
        index = int(np.searchsorted(self.starts, start, side="right") - 1)
        while written < count:
            local = int(start + written - self.starts[index])
            take = int(min(count - written, self.sizes[index] - local))
            if take <= 0:  # zero-length file
                index += 1
                continue
            if self.backend == "memmap":
                out[written : written + take] = self._map(index)[local : local + take]
            else:
                with open(self.paths[index], "rb") as handle:
                    handle.seek(local)
                    block = handle.read(take)
                if len(block) != take:
                    raise EOFError(f"{self.paths[index].name} ended early")
                out[written : written + take] = np.frombuffer(block, dtype=np.uint8)
            written += take
            index += 1
        return out

    def close(self):
        """Release the open memmaps.

        A loaded signal keeps its ``.raw`` files mapped, which on Windows stops
        them being moved or deleted. Reads after this reopen them, so it is
        safe to call at any time.
        """
        self._maps.clear()

    def __getstate__(self):
        return self.paths, self.sizes, self.backend

    def __setstate__(self, state):
        self.paths, self.sizes, self.backend = state
        self.starts = np.cumsum([0] + self.sizes)
        self._maps = {}

    def __repr__(self):
        return (
            f"<RawByteStream {self.nbytes / 1e9:.2f} GB over "
            f"{len(self.paths)} file(s), {self.backend}>"
        )


class LazyRawArray:
    """Read-on-demand array view over a :class:`RawByteStream`.

    Exposes just enough of the array protocol for dask to slice it lazily, and
    deliberately has **no** ``copy`` method: ``dask.array.from_array`` calls
    ``x.copy()`` on any array-like that offers one, which would pull the whole
    stream into RAM -- 9 GB or more per signal.
    """

    __slots__ = ("stream", "dtype", "shape")

    def __init__(self, stream, dtype, shape):
        self.stream = stream
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(s) for s in shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        if len(key) != self.ndim:
            raise IndexError(f"too many indices for shape {self.shape}")

        bounds, squeeze = [], []
        for axis, (part, size) in enumerate(zip(key, self.shape)):
            if isinstance(part, (int, np.integer)):
                index = int(part) + (size if part < 0 else 0)
                if not 0 <= index < size:
                    raise IndexError(f"index {part} out of range for axis {axis}")
                bounds.append((index, index + 1))
                squeeze.append(axis)
            elif isinstance(part, slice):
                start, stop, step = part.indices(size)
                if step != 1:
                    raise NotImplementedError("only unit-step slices are supported")
                bounds.append((start, max(stop, start)))
            else:
                raise NotImplementedError(f"unsupported index {part!r}")

        out_shape = tuple(stop - start for start, stop in bounds)
        out = np.empty(out_shape, dtype=self.dtype)
        if out.size:
            # Element strides, slowest axis first.
            strides = [1] * self.ndim
            for axis in range(self.ndim - 2, -1, -1):
                strides[axis] = strides[axis + 1] * self.shape[axis + 1]

            # Longest contiguous run: the trailing axes that are fully
            # selected, plus the selected range of the axis just before them.
            full_from = self.ndim
            while full_from > 0 and bounds[full_from - 1] == (
                0,
                self.shape[full_from - 1],
            ):
                full_from -= 1
            run_axis = max(full_from - 1, 0)
            run_start, run_stop = bounds[run_axis]
            run_elements = (run_stop - run_start) * strides[run_axis]

            flat = out.reshape(-1)
            position = 0
            itemsize = self.dtype.itemsize
            leading = [range(start, stop) for start, stop in bounds[:run_axis]]
            for combination in itertools.product(*leading):
                offset = sum(i * strides[a] for a, i in enumerate(combination))
                offset += run_start * strides[run_axis]
                raw = self.stream.read(offset * itemsize, run_elements * itemsize)
                flat[position : position + run_elements] = raw.view(self.dtype)
                position += run_elements

        if squeeze:
            out = out.reshape(
                tuple(n for axis, n in enumerate(out_shape) if axis not in squeeze)
            )
        return out

    def __getstate__(self):
        return self.stream, self.dtype, self.shape

    def __setstate__(self, state):
        self.stream, self.dtype, self.shape = state

    def __repr__(self):
        return f"<LazyRawArray {self.shape} {self.dtype.name} over {self.stream}>"


class RawFrameStore:
    """A frame-addressable view over an acquisition's ``.raw`` file(s)."""

    def __init__(self, raw_paths, frame_shape, dtype, *, strict=False, backend="memmap"):
        self.frame_shape = tuple(int(s) for s in frame_shape)
        self.dtype = np.dtype(dtype)
        self.frame_bytes = int(np.prod(self.frame_shape)) * self.dtype.itemsize

        usable = [Path(p) for p in raw_paths if Path(p).stat().st_size > 0]
        if not usable:
            raise FileNotFoundError("No usable .raw file found for this dataset")
        self.stream = RawByteStream(usable, backend=backend)

        # The frame grid is laid over the *joined* stream: a file boundary may
        # fall inside a frame, so per-file frame counts would be wrong.
        self.n_frames, remainder = divmod(self.stream.nbytes, self.frame_bytes)
        self.n_frames = int(self.n_frames)
        if remainder:
            message = (
                f"{' + '.join(p.name for p in usable)}: {self.stream.nbytes} B is "
                f"{self.n_frames} whole frames of {self.frame_bytes} B plus "
                f"{remainder} B"
            )
            if strict:
                raise ValueError(message)
            _logger.warning("%s; the partial final frame is dropped.", message)
        if self.n_frames == 0:
            raise ValueError("the raw file(s) hold less than one frame")

        straddling = [
            int(edge // self.frame_bytes)
            for edge in self.stream.starts[1:-1]
            if edge % self.frame_bytes
        ]
        if straddling:
            _logger.info(
                "%d frames over %d files; frame(s) %s straddle a file boundary "
                "(handled: the files are read as one byte stream)",
                self.n_frames,
                len(usable),
                ", ".join(str(f) for f in straddling),
            )

    @property
    def paths(self):
        return self.stream.paths

    @property
    def nbytes(self):
        return self.n_frames * self.frame_bytes

    def locate(self, frame):
        """Map a global frame index to ``(file_index, byte offset in it)``.

        A frame may continue into the next file; this reports where it starts.
        """
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} out of range (0..{self.n_frames - 1})")
        start = frame * self.frame_bytes
        index = int(np.searchsorted(self.stream.starts, start, side="right") - 1)
        return index, int(start - self.stream.starts[index])

    def read_frame(self, frame):
        """Read a single frame eagerly, as ``frame_shape``.  Handy for checks."""
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} out of range (0..{self.n_frames - 1})")
        raw = self.stream.read(frame * self.frame_bytes, self.frame_bytes)
        return raw.view(self.dtype).reshape(self.frame_shape)

    def as_dask(self, chunk_mb=DEFAULT_CHUNK_MB, backend=None):
        """Return the series as a dask array of shape ``(n_frames, *frame_shape)``."""
        if backend is not None and backend != self.stream.backend:
            self.stream.backend = backend
        frames_per_chunk, inner_chunks = _pick_chunks(
            self.frame_shape, self.dtype.itemsize, chunk_mb, self.n_frames
        )
        chunks = (frames_per_chunk,) + inner_chunks
        _logger.info(
            "chunking %d frames of %s as %s = %.1f MB per chunk",
            self.n_frames,
            self.frame_shape,
            chunks,
            int(np.prod(chunks)) * self.dtype.itemsize / 1024**2,
        )

        shape = (self.n_frames,) + self.frame_shape
        # Name the graph node ourselves: letting dask tokenize the array would
        # hash gigabytes of mapped data on every call.
        name = "gatan-raw-" + stable_hash(
            [(p.resolve(), s) for p, s in zip(self.stream.paths, self.stream.sizes)],
            shape,
            self.dtype.str,
            chunks,
        )
        proxy = LazyRawArray(self.stream, self.dtype, shape)
        return da.from_array(proxy, chunks=chunks, name=name)

    def __repr__(self):
        return (
            f"<RawFrameStore {self.n_frames} frames of {self.frame_shape} "
            f"{self.dtype.name}, {self.nbytes / 1e9:.2f} GB "
            f"over {len(self.stream.paths)} file(s)>"
        )
