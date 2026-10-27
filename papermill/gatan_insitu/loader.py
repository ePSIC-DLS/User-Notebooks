"""Lazily load Gatan multiframe (in-situ) STEM/EELS acquisitions into HyperSpy.

Entry points
------------
:func:`load_insitu_signal`
    One ``.dm4`` + its ``.raw`` companion(s) -> one calibrated lazy signal.
:func:`load_insitu_dataset`
    A whole acquisition folder -> ``{short name: signal}``.
:func:`save_rechunked`
    Write a spectrum-contiguous copy, for workflows that read pixel by pixel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import hyperspy.api as hs
import numpy as np

from .dm_tags import _get, read_raw_format_info
from .raw_store import DEFAULT_CHUNK_MB, RawFrameStore

_logger = logging.getLogger(__name__)

__all__ = [
    "load_insitu_signal",
    "load_insitu_dataset",
    "save_rechunked",
    "describe_folder",
]


def _as_slice(frames, n_frames):
    if frames is None:
        return slice(None)
    if isinstance(frames, slice):
        return frames
    if isinstance(frames, int):
        return slice(0, min(frames, n_frames))
    start, stop = frames
    return slice(start, stop)


def _time_axis(info, n_frames, frame_slice, time_origin, uniform):
    """Build the navigation axis that runs over frames."""
    times = info.frame_times
    if times is not None and times.size >= n_frames:
        times = np.asarray(times[:n_frames], dtype=float)[frame_slice]
    else:
        times = None

    if times is None or times.size == 0:
        return {
            "name": "Frame",
            "size": n_frames,
            "offset": 0.0,
            "scale": 1.0,
            "units": "frame",
            "navigate": True,
        }

    if time_origin == "first_frame":
        times = times - times[0]
    elif time_origin != "experiment_start":
        raise ValueError(
            "time_origin must be 'experiment_start' or 'first_frame', "
            f"got {time_origin!r}"
        )

    if not uniform and times.size > 1:
        return {"axis": times, "name": "Time", "units": "s", "navigate": True}

    scale = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    return {
        "name": "Time",
        "size": int(times.size),
        "offset": float(times[0]),
        "scale": scale,
        "units": "s",
        "navigate": True,
    }


def _metadata(info, title, n_frames, frame_times):
    """Assemble HyperSpy metadata from the DM tag tree."""
    tags = info.image_tags
    metadata = {
        "General": {
            "title": title,
            "original_filename": info.dm4_path.name,
        },
        "Signal": {"quantity": info.intensity_units or "Intensity"},
    }
    if info.signal_type:
        metadata["Signal"]["signal_type"] = info.signal_type

    date = _get(tags, "SI.Acquisition.Date")
    start = _get(tags, "In-situ.Recorded.Experiment Start Time")
    if start:
        text = str(start)
        metadata["General"]["date"] = text.split(" ")[0]
        if " " in text:
            metadata["General"]["time"] = text.split(" ", 1)[1]
    elif date:
        metadata["General"]["date"] = str(date)

    tem = {}
    voltage = _get(tags, "Microscope Info.Voltage")
    if voltage:
        tem["beam_energy"] = float(voltage) / 1e3  # HyperSpy wants kV
    camera_length = _get(tags, "Microscope Info.STEM Camera Length")
    if camera_length is not None:
        tem["camera_length"] = float(camera_length)
    magnification = _get(tags, "Microscope Info.Indicated Magnification")
    if magnification is not None:
        tem["magnification"] = float(magnification)

    convergence = _get(tags, "EELS.Experimental Conditions.Convergence semi-angle (mrad)")
    if convergence is not None:
        tem["convergence_angle"] = float(convergence)
    collection = _get(tags, "EELS.Experimental Conditions.Collection semi-angle (mrad)")
    if collection is not None:
        tem["Detector"] = {"EELS": {"collection_angle": float(collection)}}

    dwell = _get(tags, "SI.Acquisition.Pixel time (s)")
    if dwell is not None:
        tem["dwell_time"] = float(dwell)
    exposure = _get(tags, "EELS.Acquisition.Exposure (s)")
    if exposure is not None:
        tem.setdefault("Detector", {}).setdefault("EELS", {})
        tem["Detector"]["EELS"]["exposure"] = float(exposure)
    spectrometer = _get(tags, "EELS Spectrometer.Instrument name")
    if spectrometer:
        tem.setdefault("Detector", {}).setdefault("EELS", {})
        tem["Detector"]["EELS"]["spectrometer"] = str(spectrometer)

    if tem:
        metadata["Acquisition_instrument"] = {"TEM": tem}

    # Namespaced node for the things that are specific to a multiframe record.
    insitu = {
        "n_frames": int(n_frames),
        "raw_files": [str(p) for p in info.raw_paths],
        "n_frames_declared": int(info.n_frames_declared),
        "raw_frame_layout": "+".join(a["name"] for a in info.frame_axes),
    }
    if frame_times is not None and frame_times.size:
        insitu["frame_times_s"] = np.asarray(frame_times, dtype=float)
        if frame_times.size > 1:
            insitu["frame_interval_s"] = float(np.median(np.diff(frame_times)))
    experiment_start = _get(tags, "In-situ.Recorded.Experiment Start Time")
    if experiment_start:
        insitu["experiment_start"] = str(experiment_start)
    duration = _get(tags, "In-situ.Recorded.Duration (s)")
    if duration is not None:
        insitu["recorded_duration_s"] = float(duration)
    metadata["InSitu"] = insitu

    return metadata


def load_insitu_signal(
    dm4_path,
    *,
    chunk_mb=DEFAULT_CHUNK_MB,
    layout="spectrum",
    frames=None,
    time_origin="experiment_start",
    uniform_time=True,
    raw_paths=None,
    backend="memmap",
    strict=False,
    lazy=True,
):
    """Load one multiframe acquisition as a calibrated lazy HyperSpy signal.

    Parameters
    ----------
    dm4_path : str or Path
        The ``.dm4`` of the acquisition.  Its ``.raw`` companions are found
        automatically; if the file is not a multiframe record it is loaded
        normally with :func:`hyperspy.api.load`.
    chunk_mb : float
        Target dask chunk size.  Chunks are cut along the axis that is slowest
        in the file, so they stay contiguous on disk.
    layout : {'spectrum', 'native'}
        ``'spectrum'`` (default) gives the familiar ``(time, y, x | E)``
        ``EELSSpectrum``.  ``'native'`` keeps the on-disk ``(time, E | y, x)``
        order as a ``Signal2D``, where every navigation step is an
        energy-filtered image -- no transpose, so chunk reads map one-to-one
        onto contiguous disk reads.
    frames : int, slice or (start, stop), optional
        Restrict to a subset of frames, e.g. ``frames=5`` for a quick look.
    time_origin : {'experiment_start', 'first_frame'}
        Whether the time axis counts from the start of the in-situ experiment
        (as recorded by DM) or from the first loaded frame.
    uniform_time : bool
        ``True`` fits a uniform axis to the per-frame timestamps (they are
        regular to well under a percent in practice) and keeps the exact values
        in ``metadata.InSitu.frame_times_s``.  ``False`` builds a non-uniform
        axis from the timestamps themselves.
    raw_paths : sequence of path, optional
        Override raw-file discovery, in frame order.
    backend : {'memmap', 'fromfile'}
        ``'memmap'`` lets the OS page the data in and is fastest on local
        disks.  ``'fromfile'`` issues explicit seeks and reads instead, which
        is more predictable over a network share.
    strict : bool
        Raise instead of warning when a raw file does not hold a whole number
        of frames (an aborted recording).
    lazy : bool
        Set ``False`` to compute the array immediately.  Only sensible for a
        small ``frames`` subset -- the full high-loss stream here is 9 GB.
    """
    dm4_path = Path(dm4_path)
    info = read_raw_format_info(dm4_path, raw_paths=raw_paths)
    if info is None:
        _logger.info("%s is not a multiframe record; loading it directly.", dm4_path.name)
        return hs.load(dm4_path, lazy=lazy)
    if not info.raw_paths:
        _logger.warning(
            "%s declares a raw stream but no .raw file was found; "
            "loading the single frame stored in the .dm4.",
            dm4_path.name,
        )
        return hs.load(dm4_path, lazy=lazy)

    store = RawFrameStore(
        info.raw_paths, info.frame_shape, info.dtype, strict=strict, backend=backend
    )
    if info.n_frames_declared and store.n_frames != info.n_frames_declared:
        _logger.warning(
            "%s: the tags declare %d frames but the raw file(s) hold %d; "
            "using the file contents.",
            dm4_path.name,
            info.n_frames_declared,
            store.n_frames,
        )

    data = store.as_dask(chunk_mb=chunk_mb)

    frame_slice = _as_slice(frames, store.n_frames)
    if frame_slice != slice(None):
        data = data[frame_slice]
    n_frames = data.shape[0]

    frame_axes = [dict(axis, navigate=True) for axis in info.frame_axes]
    time_axis = _time_axis(info, store.n_frames, frame_slice, time_origin, uniform_time)

    is_si = info.is_spectrum_image
    if layout not in ("spectrum", "native"):
        raise ValueError(f"layout must be 'spectrum' or 'native', got {layout!r}")

    if is_si and layout == "spectrum":
        # (time, E, y, x) -> (time, y, x, E).  Lazy: the graph transposes the
        # chunk order, the underlying reads stay contiguous.
        data = data.transpose(0, 2, 3, 1)
        energy, spatial = frame_axes[0], frame_axes[1:]
        energy["navigate"] = False
        axes = [time_axis] + spatial + [energy]
        signal = hs.signals.Signal1D(data, axes=axes)
    elif is_si:
        for axis in frame_axes[1:]:
            axis["navigate"] = False
        axes = [time_axis] + frame_axes
        signal = hs.signals.Signal2D(data, axes=axes)
    else:
        for axis in frame_axes:
            axis["navigate"] = False
        axes = [time_axis] + frame_axes
        signal = hs.signals.Signal2D(data, axes=axes)

    signal = signal.as_lazy()
    if not isinstance(signal.data, da.Array):  # pragma: no cover - guards a regression
        raise RuntimeError("the signal was computed eagerly; chunking is broken")

    title = dm4_path.stem
    times = info.frame_times
    if times is not None and times.size >= store.n_frames:
        times = np.asarray(times[: store.n_frames])[frame_slice]
    metadata = _metadata(info, title, n_frames, times)
    for key, value in _flatten(metadata):
        signal.metadata.set_item(key, value)
    signal.original_metadata.add_dictionary(info.tags)

    if info.signal_type == "EELS" and layout == "spectrum" and is_si:
        signal.set_signal_type("EELS")
        tem = signal.metadata.get_item("Acquisition_instrument.TEM", None)
        if tem is not None:
            signal.set_microscope_parameters(
                beam_energy=tem.get_item("beam_energy", None),
                convergence_angle=tem.get_item("convergence_angle", None),
                collection_angle=tem.get_item(
                    "Detector.EELS.collection_angle", None
                ),
            )

    if not lazy:
        signal.compute()
    return signal


def _flatten(mapping, prefix=""):
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, path)
        else:
            yield path, value


def _short_name(stem):
    """``'STEM SI_EELS HL SI'`` -> ``'EELS HL'``."""
    name = stem
    for prefix in ("STEM SI_", "SI_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.endswith(" SI"):
        name = name[: -len(" SI")]
    return name.strip() or stem


def describe_folder(folder):
    """Report what is in an acquisition folder without loading anything."""
    folder = Path(folder)
    lines = []
    for dm4 in sorted(folder.glob("*.dm4")):
        info = read_raw_format_info(dm4)
        if info is None:
            size = dm4.stat().st_size / 1e6
            lines.append(f"{dm4.name}: single-frame .dm4, {size:.1f} MB")
        else:
            lines.append(info.summary())
    return "\n".join(lines)


def load_insitu_dataset(folder, *, include_single_frame=True, **kwargs):
    """Load every acquisition in ``folder`` into ``{short name: signal}``.

    ``include_single_frame`` controls whether plain ``.dm4`` files with no raw
    companion (the SI survey image, typically) are included.
    """
    folder = Path(folder)
    signals = {}
    for dm4 in sorted(folder.glob("*.dm4")):
        info = read_raw_format_info(dm4)
        if info is None and not include_single_frame:
            continue
        key = _short_name(dm4.stem)
        if key in signals:
            key = dm4.stem
        try:
            signals[key] = load_insitu_signal(dm4, **kwargs)
        except Exception as error:  # keep going; report the rest
            _logger.error("Could not load %s: %s", dm4.name, error)
    return signals


def _save_chunks(signal, target_mb):
    """Spectrum-contiguous chunks: whole signal axes, a compact nav tile.

    The navigation budget is spread *evenly* over the spatial axes rather than
    filling one axis first, so a chunk covers a square-ish patch of the scan.
    That keeps a region of interest inside few chunks. The slowest navigation
    axis (time, for a multiframe record) is kept at one frame per chunk, since
    frames are usually looked at one at a time.
    """
    shape = signal.data.shape
    n_sig = signal.axes_manager.signal_dimension
    n_nav = len(shape) - n_sig

    sig_elements = int(np.prod(shape[n_nav:])) or 1
    budget = max(1, int(target_mb * 1024 * 1024) // signal.data.dtype.itemsize)
    nav_budget = max(1, budget // sig_elements)

    chunks = [1] * n_nav + list(shape[n_nav:])
    if n_nav == 0:
        return tuple(chunks)

    # Axis 0 is time when there is more than one navigation axis; spread the
    # budget over the rest. With a single nav axis, it takes the whole budget.
    spread = list(range(1, n_nav)) if n_nav > 1 else [0]
    side = max(1, int(round(nav_budget ** (1.0 / len(spread)))))
    for axis in spread:
        chunks[axis] = int(min(shape[axis], side))

    # Use up any slack left by the per-axis caps.
    for axis in spread:
        room = max(1, nav_budget // max(int(np.prod([chunks[a] for a in spread])) // chunks[axis], 1))
        chunks[axis] = int(min(shape[axis], room))
    return tuple(chunks)


def save_rechunked(signal, path, *, target_mb=64, chunks=None, overwrite=None, **kwargs):
    """Write ``signal`` to ``.hspy`` as a compact, spectrum-contiguous copy.

    Use this for **size and portability**, not for speed.  Cropping in energy
    and frames turns the 9 GB ``.raw`` + ``.dm4`` pair into one self-contained
    file (gzipped by default, ~16x on this data), which is what you want for
    sharing, archiving, or working away from the original storage.

    It will *not* make per-pixel work faster.  Measured on this data the raw
    energy-sliced stream beats a spectrum-contiguous ``.hspy`` copy by 2-3x on
    every access pattern tried, including ``remove_background``, because each
    raw chunk already holds complete ``y`` and ``x`` planes -- so assembling
    whole spectra only concatenates energy chunks within a frame, which is
    cheap.  Pass ``compression=None`` for a faster but much larger file.

    Chunks are always passed explicitly: HDF5 rejects the dask ``-1``
    shorthand, and ``.zspy`` is broken in this environment (rosettasciio 0.9
    calls ``zarr.storage.NestedDirectoryStore``, removed in zarr 3).
    """
    path = Path(path)
    if path.suffix.lower() == ".zspy":
        raise ValueError(
            "This environment cannot write .zspy (rosettasciio 0.9 + zarr 3). "
            "Use .hspy, or upgrade rosettasciio / pin zarr<3 first."
        )
    if path.suffix.lower() != ".hspy":
        path = path.with_suffix(".hspy")

    chunks = tuple(chunks) if chunks is not None else _save_chunks(signal, target_mb)
    nbytes = int(np.prod(chunks)) * signal.data.dtype.itemsize
    _logger.info("saving %s with chunks %s (%.1f MB)", path.name, chunks, nbytes / 1024**2)

    # deepcopy() on a lazy signal copies the graph, not the data, and carries
    # the axes and metadata across; only .data needs replacing.
    rechunked = signal.deepcopy()
    rechunked.data = signal.data.rechunk(chunks)
    rechunked.save(path, chunks=chunks, overwrite=overwrite, **kwargs)
    return path
