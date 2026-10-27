"""Read the parts of a Gatan DM4 header that describe its companion ``.raw`` stream.

A multiframe (in-situ) acquisition in GMS writes two things side by side:

* ``<name>.dm4`` -- a normal DM file holding *one* frame of the series (the last
  one) plus the complete tag tree, and
* ``<name>.raw`` -- the headerless concatenation of *every* frame, optionally
  split over several files when the recording is long.

The tag subtree ``ImageTags.In-situ.Raw File Format Info`` describes the raw
stream exactly: per-frame dimensions, element type and size, frame size in
bytes, and the calibration of each axis *already expressed in physical units*
(unlike ``ImageData.Calibrations``, which uses DM's index-origin convention).
``ImageTags.In-situ.Recorded`` adds the frame count and a per-frame timestamp.

This module turns that into a :class:`RawFormatInfo`.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rsciio.digitalmicrograph._api import DigitalMicrographReader

_logger = logging.getLogger(__name__)

#: DM image data-type code -> numpy dtype.  Mirrors
#: ``rsciio.digitalmicrograph._api.ImageObject.dtype`` so that data read from
#: the ``.raw`` file is interpreted exactly as rsciio interprets the frame
#: embedded in the ``.dm4``.  Note that code 2 is ``float32`` here -- it is
#: *not* the DM *tag* type table, where 2 means int16.
DM_IMAGE_DTYPES = {
    1: "int16",
    2: "float32",
    3: "complex64",
    5: "float32",  # 8-byte packed complex (FFT) -- read as pairs of float32
    6: "uint8",
    7: "int32",
    9: "int8",
    10: "uint16",
    11: "uint32",
    12: "float64",
    13: "complex128",
    14: "bool",
}

_RAW_SUFFIX = re.compile(r"\.raw$", re.IGNORECASE)


def _get(mapping, path, default=None):
    """Fetch a dotted key path out of DM's nested tag dictionaries."""
    current = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def read_dm_tags(path):
    """Parse a DM3/DM4 file's tag tree without reading its image data."""
    with open(path, "rb") as handle:
        reader = DigitalMicrographReader(handle)
        reader.parse_file()
        return reader.tags_dict, reader.endian or "little"


def _taggroup_order(name):
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else 0


def _find_insitu_group(tags):
    """Return the ``ImageList`` group that owns the in-situ raw stream.

    A multiframe DM4 usually holds two images: an RGB display thumbnail
    (``TagGroup0``) and the real data (``TagGroup1``).  Only the latter carries
    ``In-situ.Raw File Format Info``, so look for that rather than assuming an
    index.
    """
    image_list = tags.get("ImageList", {})
    for name in sorted(image_list, key=_taggroup_order):
        group = image_list[name]
        if _get(group, "ImageTags.In-situ.Raw File Format Info") is not None:
            return name, group
    return None, None


def _axis_name(units, index_from_end):
    """Name a frame axis from its units and its position in the frame."""
    unit = (units or "").strip()
    if unit == "eV":
        return "Energy loss"
    return {0: "x", 1: "y"}.get(index_from_end, f"axis-{index_from_end}")


def _frame_axes(raw_info):
    """Build the per-frame C-order shape and matching axis calibrations.

    DM lists frame dimensions as ``Data0..DataN`` with **Data0 varying
    fastest**, so the C-order shape is that list reversed.  Verified against
    the frame embedded in the ``.dm4`` for both a 2D image (``(y, x)``) and a
    3D spectrum image (``(E, y, x)`` -- i.e. the raw stream is a stack of
    energy-slice images, *not* a stack of spectra).
    """
    dimensions = raw_info.get("Dimensions", {})
    n_dim = len(dimensions)
    if n_dim == 0:
        raise ValueError("'Raw File Format Info.Dimensions' is empty")

    sizes, axes = [], []
    for i in reversed(range(n_dim)):
        size = int(dimensions[f"Data{i}"])
        calibration = _get(raw_info, f"Calibrations.dim-{i}", {}) or {}
        units = calibration.get("Unit") or ""
        units = "" if isinstance(units, (list, tuple)) else str(units)
        sizes.append(size)
        axes.append(
            {
                # DM's Data0 varies fastest, so dim index i counts back from
                # the end of the C-order shape: dim-0 is x, dim-1 is y.
                "name": _axis_name(units, i),
                "size": size,
                # 'Raw File Format Info' origins are already offsets in
                # physical units, so they are used as-is.
                "offset": float(calibration.get("Origin", 0.0) or 0.0),
                "scale": float(calibration.get("Scale", 1.0) or 1.0),
                "units": units,
            }
        )
    return tuple(sizes), axes


def _frame_times(recorded, n_frames):
    """Decode ``Recorded.Sync.Per Frame.Time Since Start (ns)`` into seconds.

    DM stores the 64-bit nanosecond counters as a flat array of 32-bit words --
    low word first -- so the array is twice as long as the frame count.  Older
    files sometimes store one value per frame instead, so both are accepted and
    the result is sanity-checked for monotonicity.
    """
    values = _get(recorded, "Sync.Per Frame.Time Since Start (ns)")
    if values is None:
        return None

    words = np.asarray(values).ravel()
    if words.size == 0:
        return None

    if n_frames and words.size == 2 * n_frames:
        words = words.astype(np.uint32, copy=False)
        low = words[0::2].astype(np.uint64)
        high = words[1::2].astype(np.uint64)
        nanoseconds = low | (high << np.uint64(32))
    elif n_frames and words.size == n_frames:
        nanoseconds = words.astype(np.uint64)
    else:
        _logger.warning(
            "Per-frame timestamps have %d entries, which matches neither %s "
            "nor %s frames; ignoring them.",
            words.size,
            n_frames,
            2 * (n_frames or 0),
        )
        return None

    seconds = nanoseconds.astype(np.float64) / 1e9
    if seconds.size > 1 and not np.all(np.diff(seconds) > 0):
        _logger.warning("Per-frame timestamps are not increasing; ignoring them.")
        return None
    return seconds


def find_raw_files(dm4_path, dm4_dir_listing=None):
    """Locate every ``.raw`` file belonging to ``dm4_path``, in frame order.

    Long recordings are split across files, and the naming convention varies
    between GMS versions, so three spellings are accepted::

        <stem>.raw                  <stem>.raw_1   <stem>.raw.2
        <stem>_1.raw

    The unnumbered file always sorts first.  A candidate that has its own
    ``.dm4`` beside it belongs to a different acquisition and is skipped, which
    keeps ``"<stem>_1.raw"`` from being stolen from a dataset actually named
    ``"<stem>_1"``.
    """
    dm4_path = Path(dm4_path)
    folder = dm4_path.parent
    stem = re.escape(dm4_path.stem)

    patterns = (
        re.compile(rf"^{stem}\.raw$", re.IGNORECASE),
        re.compile(rf"^{stem}\.raw[._-](\d+)$", re.IGNORECASE),
        re.compile(rf"^{stem}[._-](\d+)\.raw$", re.IGNORECASE),
    )

    listing = dm4_dir_listing if dm4_dir_listing is not None else folder.iterdir()
    found = []
    for candidate in listing:
        if not candidate.is_file():
            continue
        for pattern in patterns:
            match = pattern.match(candidate.name)
            if match is None:
                continue
            index = int(match.group(1)) if match.groups() else 0
            # Guard against adopting a sibling acquisition's raw file.
            own_dm4 = candidate.with_suffix("").with_suffix(".dm4")
            if (
                index
                and _RAW_SUFFIX.search(candidate.name)
                and own_dm4.exists()
                and own_dm4 != dm4_path
            ):
                _logger.debug("Skipping %s: it has its own .dm4", candidate.name)
                break
            found.append((index, candidate.name, candidate))
            break

    found.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in found]


@dataclass
class RawFormatInfo:
    """Everything needed to open a multiframe ``.raw`` stream lazily."""

    dm4_path: Path
    raw_paths: list
    frame_shape: tuple
    """Per-frame C-order shape: ``(y, x)`` for images, ``(E, y, x)`` for SIs."""
    frame_axes: list
    """Calibration dicts matching ``frame_shape``, slowest axis first."""
    dtype: np.dtype
    frame_bytes: int
    n_frames_declared: int
    frame_times: np.ndarray | None
    signal_type: str
    intensity_units: str
    endian: str
    tags: dict = field(repr=False)
    image_tags: dict = field(repr=False)

    @property
    def is_spectrum_image(self):
        return len(self.frame_shape) == 3 and self.frame_axes[0]["units"] == "eV"

    @property
    def frame_interval(self):
        """Median frame-to-frame interval in seconds, or ``None``."""
        if self.frame_times is None or self.frame_times.size < 2:
            return None
        return float(np.median(np.diff(self.frame_times)))

    def summary(self):
        axes = ", ".join(
            f"{a['name']}:{a['size']}"
            + (f" @{a['scale']:.4g}{a['units']}" if a["units"] else "")
            for a in self.frame_axes
        )
        interval = self.frame_interval
        timing = f", dt~{interval:.3f}s" if interval else ""
        return (
            f"{self.dm4_path.name}: {self.n_frames_declared} frames x ({axes}) "
            f"{self.dtype}, {len(self.raw_paths)} raw file(s), "
            f"{self.frame_bytes * self.n_frames_declared / 1e9:.2f} GB{timing}"
        )


def read_raw_format_info(dm4_path, raw_paths=None):
    """Inspect ``dm4_path`` and describe its companion raw stream.

    Returns ``None`` if the file is not a multiframe acquisition (no
    ``In-situ.Raw File Format Info`` tag), so callers can fall back to loading
    the ``.dm4`` normally.
    """
    dm4_path = Path(dm4_path)
    tags, endian = read_dm_tags(dm4_path)

    group_name, group = _find_insitu_group(tags)
    if group is None:
        return None

    image_tags = group.get("ImageTags", {})
    raw_info = _get(image_tags, "In-situ.Raw File Format Info")
    recorded = _get(image_tags, "In-situ.Recorded", {}) or {}

    frame_shape, frame_axes = _frame_axes(raw_info)

    type_code = int(raw_info["Data Type"])
    if type_code not in DM_IMAGE_DTYPES:
        raise NotImplementedError(f"Unsupported DM image data type {type_code}")
    dtype = np.dtype(DM_IMAGE_DTYPES[type_code])
    dtype = dtype.newbyteorder("<" if endian == "little" else ">")

    item_size = int(raw_info.get("Data Size (bytes)", dtype.itemsize))
    if item_size != dtype.itemsize:
        raise ValueError(
            f"{dm4_path.name}: DM data type {type_code} implies {dtype.itemsize} "
            f"bytes per element but the tags say {item_size}"
        )

    frame_bytes = int(raw_info.get("Frame Size (bytes)", 0)) or (
        int(np.prod(frame_shape)) * dtype.itemsize
    )
    expected = int(np.prod(frame_shape)) * dtype.itemsize
    if frame_bytes != expected:
        raise ValueError(
            f"{dm4_path.name}: tagged frame size {frame_bytes} B disagrees with "
            f"{frame_shape} x {dtype} = {expected} B"
        )

    n_declared = int(recorded.get("# Frames", 0) or 0)
    times = _frame_times(recorded, n_declared)

    if raw_paths is None:
        raw_paths = find_raw_files(dm4_path)
    else:
        raw_paths = [Path(p) for p in raw_paths]

    signal_type = ""
    if _get(image_tags, "Meta Data.Signal") == "EELS" or "EELS" in image_tags:
        signal_type = "EELS"
    elif _get(image_tags, "Meta Data.Signal") == "X-ray":
        signal_type = "EDS_TEM"

    intensity_units = _get(raw_info, "Calibrations.intensity.Unit") or ""
    if isinstance(intensity_units, (list, tuple)):
        intensity_units = ""

    _logger.debug("%s: in-situ data in ImageList.%s", dm4_path.name, group_name)

    return RawFormatInfo(
        dm4_path=dm4_path,
        raw_paths=raw_paths,
        frame_shape=frame_shape,
        frame_axes=frame_axes,
        dtype=dtype,
        frame_bytes=frame_bytes,
        n_frames_declared=n_declared,
        frame_times=times,
        signal_type=signal_type,
        intensity_units=str(intensity_units),
        endian=endian,
        tags=tags,
        image_tags=image_tags,
    )


def stable_hash(*parts):
    """A short deterministic hash, used to name dask graph nodes."""
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
