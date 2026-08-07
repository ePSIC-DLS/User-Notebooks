"""Lazy loading of Gatan OneView in-situ movies stored as nested frame folders.

A OneView in-situ recording is written as one .dm4/.dm3 file per frame, nested as
``<Overview>/Hour_HH/Minute_MM/Second_SS/*.dm4``.  Those per-frame files carry no
calibration (pixel scale 1, no units).  All of the useful metadata -- pixel size,
frame rate, exposure, high tension, magnification, in-situ holder channels -- lives
in the single root file written alongside the folders, ``<Overview>/<Overview>.dm4``.

``load_oneview_insitu`` globs the frames, stacks them lazily into one Signal2D and
transplants the root file's calibration and metadata onto the stack.

    >>> from oneview_insitu import load_oneview_insitu
    >>> s = load_oneview_insitu("/path/to/AuNi_SWCNT/Overview28/Overview28.dm4")
    >>> s
    <LazySignal2D, title: Overview28, dimensions: (3079|2048, 2048)>
"""

from __future__ import annotations

import gc
import re
import warnings
from pathlib import Path

import numpy as np
import hyperspy.api as hs

__all__ = ["load_oneview_insitu", "find_frame_files", "read_root_metadata",
           "raise_open_file_limit", "blockwise_mean"]

# Everything of interest in the root file hangs off this node.
_TAGS = "ImageList.TagGroup0"

_FRAME_GLOBS = ("Hour_*/Minute_*/Second_*/*{ext}", "**/*{ext}")


def raise_open_file_limit(target=None, verbose=True):
    """Raise the process's open-file soft limit to the hard limit.

    Recent RosettaSciIO reads lazy DM files through :func:`numpy.memmap`
    (``rsciio.utils._distributed.slice_memmap``).  Every live memmap holds a file
    descriptor, so a reduction across N frames can need N descriptors at once --
    far above the usual 1024/4096 soft limit, giving
    ``OSError: [Errno 24] Too many open files``.

    Call this once, before loading, on Linux/macOS.  A normal user may raise the
    soft limit up to the hard limit without root.  No-op on Windows, which has no
    ``resource`` module.

    Returns the new soft limit, or None if it could not be changed.
    """
    try:
        import resource
    except ImportError:  # Windows
        if verbose:
            print("raise_open_file_limit: no 'resource' module (Windows); skipped")
        return None

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY:
        hard = 1_048_576
    new = hard if target is None else min(target, hard)
    if new <= soft:
        if verbose:
            print(f"open-file limit already {soft} (hard {hard}); unchanged")
        return soft
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new, hard))
    except (ValueError, OSError) as err:
        warnings.warn(f"Could not raise the open-file limit from {soft}: {err}",
                      stacklevel=2)
        return None
    if verbose:
        print(f"open-file limit raised {soft} -> {new} (hard {hard})")
    return new


def blockwise_mean(signal, block_size=100, dtype=np.float64, verbose=True):
    """Mean over the navigation axis, computed a block of frames at a time.

    ``signal.mean(axis='time').compute()`` builds one dask task per frame and lets
    the scheduler hold thousands of memmap-backed results alive at once, which is
    what exhausts the file-descriptor limit on big datasets.  Accumulating block by
    block bounds the number of simultaneously open frames to *block_size*.

    Returns an eager Signal2D carrying the input's signal-axis calibration.
    """
    data = signal.data
    n = data.shape[0]
    acc = np.zeros(data.shape[1:], dtype=dtype)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        acc += data[start:stop].compute().sum(axis=0, dtype=dtype)
        # Drop the block's memmaps before opening the next block's.
        gc.collect()
        if verbose:
            print(f"\r  mean: {stop}/{n} frames", end="")
    if verbose:
        print()

    out = hs.signals.Signal2D(acc / n)
    for axis, ref in zip(out.axes_manager.signal_axes,
                         signal.axes_manager.signal_axes):
        axis.scale, axis.units = ref.scale, ref.units
        axis.offset, axis.name = ref.offset, ref.name
    out.metadata.add_dictionary(signal.metadata.as_dictionary())
    out.metadata.set_item("General.title",
                          f"{signal.metadata.General.title} mean")
    return out


def _natural_key(path):
    """Sort key that orders Frame_9 before Frame_10 regardless of zero padding."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", str(path))]


def find_frame_files(data_dir, extension=None):
    """Return the per-frame files under *data_dir*, in acquisition order.

    Tries the Hour/Minute/Second layout first and falls back to a recursive
    search.  *extension* defaults to whichever of .dm4/.dm3 is present.
    """
    data_dir = Path(data_dir)
    extensions = [extension] if extension else [".dm4", ".dm3"]

    for ext in extensions:
        for pattern in _FRAME_GLOBS:
            files = [p for p in data_dir.glob(pattern.format(ext=ext)) if p.is_file()]
            if files:
                return sorted(files, key=_natural_key)
    return []


def _seconds_histogram(files):
    """Frames per Second_SS folder, keyed by the folder's parent path."""
    counts = {}
    for f in files:
        counts[f.parent] = counts.get(f.parent, 0) + 1
    return counts


def _as_dict(node):
    """Subtrees come back as DictionaryTreeBrowser; normalise to a plain dict."""
    if node is None:
        return {}
    if hasattr(node, "as_dictionary"):
        return node.as_dictionary()
    return node if isinstance(node, dict) else {}


def read_root_metadata(root_file, lazy=True):
    """Load the root .dm4 and pull out the values needed to describe the movie.

    Returns ``(root_signal, info)`` where *info* is a plain dict.
    """
    root = hs.load(str(root_file), lazy=lazy)
    if isinstance(root, list):  # defensive: some root files hold >1 image
        root = root[0]

    om = root.original_metadata

    def tag(path, default=None):
        return om.get_item(f"{_TAGS}.{path}", default)

    fps = tag("ImageTags.In-situ.Recorded.Sync.Linear.Frame Rate (fps)")
    if fps is None:
        fps = tag("ImageTags.DataBar.Frames per Second")
    exposure = tag("ImageTags.DataBar.Exposure Time (s)")
    if exposure is None:
        exposure = tag("ImageTags.Acquisition.Parameters.High Level.Exposure (s)")
    if fps is None and exposure:  # last resort: assume back-to-back exposures
        fps = 1.0 / exposure

    # Registered in-situ holder channels (Wildfire temperature, power, ...)
    channels = []
    reg = _as_dict(tag("ImageTags.DataBar.Registered elements"))
    for key in sorted(reg):
        entry = _as_dict(reg[key])
        if "Path" in entry:
            channels.append({"name": entry.get("Name"),
                             "path": entry.get("Path"),
                             "units": entry.get("Units")})

    info = {
        "title": tag("Name") or Path(root_file).stem,
        "n_frames": tag("ImageTags.In-situ.Recorded.# Frames"),
        "duration_s": tag("ImageTags.In-situ.Recorded.Duration (s)"),
        "frame_rate_fps": fps,
        "first_frame_time_s": (tag("ImageTags.In-situ.Recorded.Sync.Linear."
                                   "First Frame Time (ns)", 0) or 0) * 1e-9,
        "experiment_start": tag("ImageTags.In-situ.Recorded.Experiment Start Time"),
        "exposure_s": exposure,
        "beam_energy_kV": (tag("ImageTags.Microscope Info.Voltage", 0) or 0) / 1000.0,
        "magnification": tag("ImageTags.Microscope Info.Indicated Magnification"),
        "actual_magnification": tag("ImageTags.Microscope Info.Actual Magnification"),
        "camera": (tag("ImageTags.Acquisition.Device.Name")
                   or tag("ImageTags.DataBar.Device Name")),
        "camera_id": tag("ImageTags.Acquisition.Device.Source ID"),
        "binning": tag("ImageTags.DataBar.Binning"),
        "detector_temperature_C": tag("ImageTags.Acquisition.Device.Temperature (C)"),
        # counts -> electrons conversion stored by DigitalMicrograph
        "intensity_scale": tag("ImageData.Calibrations.Brightness.Scale"),
        "intensity_units": tag("ImageData.Calibrations.Brightness.Units"),
        "insitu_channels": channels,
    }
    return root, info


def _split_start_time(stamp):
    """'2026-07-02 17:00:37.312' -> ('2026-07-02', '17:00:37.312')."""
    if not stamp:
        return None, None
    parts = str(stamp).split(" ")
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(stamp), None


def load_oneview_insitu(root_file, data_dir=None, extension=None, lazy=True,
                        apply_gain=False, new_axis_name="time", check=True,
                        verbose=True):
    """Lazily load a nested OneView in-situ movie as a single calibrated signal.

    Parameters
    ----------
    root_file : str or Path
        The single ``OverviewNN.dm4`` written next to the Hour_* folders; it is
        the source of all calibration and metadata.
    data_dir : str or Path, optional
        Folder holding the ``Hour_*`` tree.  Defaults to ``root_file``'s folder.
    extension : str, optional
        ``'.dm4'`` or ``'.dm3'``.  Auto-detected by default.
    lazy : bool
        Keep frames on disk (default).  ``False`` reads everything into memory.
    apply_gain : bool
        Multiply by the DM brightness scale to convert counts to electrons.
        Stays lazy, but promotes the data to float32.
    new_axis_name : str
        Name of the stacking (time) navigation axis.
    check : bool
        Warn when the frame count or per-second frame counts disagree with the
        metadata, which indicates dropped frames or an incomplete transfer.

    Returns
    -------
    hyperspy.signals.Signal2D
        Lazy when ``lazy=True``, with a calibrated time navigation axis.
    """
    root_file = Path(root_file)
    data_dir = Path(data_dir) if data_dir is not None else root_file.parent

    root, info = read_root_metadata(root_file, lazy=True)

    files = find_frame_files(data_dir, extension)
    if not files:
        raise FileNotFoundError(
            f"No frame files found under {data_dir}. Expected a layout like "
            f"Hour_00/Minute_01/Second_05/*.dm4"
        )
    if verbose:
        print(f"Found {len(files)} frames under {data_dir}")

    if check:
        expected = info["n_frames"]
        if expected and len(files) != expected:
            warnings.warn(
                f"Metadata reports {expected} frames but {len(files)} files were "
                f"found ({expected - len(files):+d}); frames may be missing.",
                stacklevel=2,
            )
        fps = info["frame_rate_fps"]
        if fps:
            hist = _seconds_histogram(files)
            # The first and last second of a recording are legitimately partial.
            interior = sorted(hist)[1:-1] if len(hist) > 2 else []
            odd = [p for p in interior if hist[p] != round(fps)]
            if odd:
                warnings.warn(
                    f"{len(odd)} of {len(interior)} interior Second_* folders do not "
                    f"hold {round(fps)} frames (e.g. {odd[0].name}: {hist[odd[0]]}). "
                    f"The uniform time axis will drift where frames were dropped.",
                    stacklevel=2,
                )

    # stack_metadata=False keeps the per-frame tag trees out of the stack; the
    # root file's metadata is attached below instead. With thousands of frames
    # the default (True) is both slow and enormous.
    s = hs.load([str(f) for f in files], lazy=lazy, stack=True,
                stack_metadata=False, new_axis_name=new_axis_name,
                show_progressbar=verbose)

    _apply_signal_calibration(s, root, verbose=verbose)
    _apply_time_axis(s, info, new_axis_name)
    _apply_metadata(s, root, info, root_file, len(files))

    if apply_gain and info["intensity_scale"]:
        s.data = s.data * np.float32(info["intensity_scale"])
        s.metadata.set_item("Signal.quantity",
                            f"Intensity ({info['intensity_units'] or 'e-'})")

    if verbose:
        _print_summary(s, info)
    return s


def _apply_signal_calibration(s, root, verbose=True):
    """Copy pixel size/units from the root image onto the stack's x/y axes."""
    for axis, ref in zip(s.axes_manager.signal_axes, root.axes_manager.signal_axes):
        scale = ref.scale
        # If the root image was saved at a different binning than the frames,
        # rescale the pixel size so it still describes the frame.
        if ref.size != axis.size and axis.size:
            scale = ref.scale * ref.size / axis.size
            if verbose:
                print(f"Root image is {ref.size} px vs {axis.size} px per frame; "
                      f"pixel size rescaled to {scale:.6g} {ref.units}")
        axis.scale = scale
        axis.units = ref.units
        axis.offset = ref.offset
        axis.name = ref.name


def _apply_time_axis(s, info, new_axis_name):
    fps = info["frame_rate_fps"]
    axis = s.axes_manager[new_axis_name]
    axis.name = "time"
    if fps:
        axis.scale = 1.0 / fps
        axis.units = "s"
        axis.offset = info["first_frame_time_s"] or 0.0
    else:
        warnings.warn("No frame rate found in the root file; the time axis is "
                      "left in frame index units.", stacklevel=3)
        axis.units = "frame"


def _apply_metadata(s, root, info, root_file, n_files):
    """Map the DM tags onto HyperSpy's metadata tree, keeping the raw tags too."""
    md = s.metadata
    md.set_item("General.title", info["title"])
    md.set_item("General.original_filename", root_file.name)

    date, time = _split_start_time(info["experiment_start"])
    if date:
        md.set_item("General.date", date)
    if time:
        md.set_item("General.time", time)

    if not s.metadata.has_item("Signal.quantity"):
        md.set_item("Signal.quantity", "Intensity (counts)")

    tem = "Acquisition_instrument.TEM."
    if info["beam_energy_kV"]:
        md.set_item(tem + "beam_energy", info["beam_energy_kV"])  # kV
    if info["magnification"]:
        md.set_item(tem + "magnification", info["magnification"])
    if info["actual_magnification"]:
        md.set_item(tem + "actual_magnification", info["actual_magnification"])
    cam = tem + "Detector.Camera."
    for key, value in (("name", info["camera"]), ("id", info["camera_id"]),
                       ("exposure", info["exposure_s"]), ("binning", info["binning"]),
                       ("frame_rate", info["frame_rate_fps"]),
                       ("temperature", info["detector_temperature_C"])):
        if value is not None:
            md.set_item(cam + key, value)

    ins = "Acquisition_instrument.TEM.In_situ."
    for key in ("frame_rate_fps", "duration_s", "experiment_start"):
        if info[key] is not None:
            md.set_item(ins + key, info[key])
    md.set_item(ins + "n_frames_expected", info["n_frames"])
    md.set_item(ins + "n_frames_loaded", n_files)
    if info["insitu_channels"]:
        md.set_item(ins + "logged_channels", info["insitu_channels"])
    if info["intensity_scale"]:
        md.set_item(ins + "counts_to_electrons", info["intensity_scale"])

    # Keep the complete root tag tree so nothing is lost.
    s.original_metadata.add_dictionary(root.original_metadata.as_dictionary())


def _print_summary(s, info):
    nav = s.axes_manager.navigation_axes[0]
    sig = s.axes_manager.signal_axes[0]
    print(f"\n{s}")
    print(f"  time      : {nav.size} frames @ {info['frame_rate_fps']} fps "
          f"({nav.size * nav.scale:.2f} s)")
    print(f"  pixel size: {sig.scale:.6g} {sig.units}")
    print(f"  beam      : {info['beam_energy_kV']:g} kV, "
          f"mag {info['magnification']:g}x")
    print(f"  camera    : {info['camera']}, exposure {info['exposure_s']} s, "
          f"binning {info['binning']}")
    if info["insitu_channels"]:
        names = ", ".join(c["path"] for c in info["insitu_channels"])
        print(f"  in-situ   : {names}")
