"""Helpers for working with multiframe in-situ records in a Jupyter notebook.

Three things that bite in a notebook and are fixed here:

* **Navigators.** ``plot()`` on a lazy signal builds its navigator by summing
  over a *single* chunk, so the image you steer with shows ~315 of 3456 energy
  channels rather than the whole spectrum, and it takes a few seconds.
  :func:`make_navigator` gives a real one instead -- ideally the simultaneously
  recorded image stream (HAADF, ADF, ...), which rebins exactly onto the SI grid.
* **``compute()`` returns ``None``.** It works in place, so the obvious
  one-liner throws away the result.  :func:`computed` returns the signal.
* **First look.** :func:`quicklook` draws survey, image frames, summed spectra and the
  intensity-versus-time trend in one pass over the data, and
  :func:`summary_table` lists a folder as a DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np

from .dm_tags import read_raw_format_info
from .loader import _short_name

_logger = logging.getLogger(__name__)

__all__ = [
    "classify_signals",
    "computed",
    "frame_range",
    "image_streams",
    "save_frame_gif",
    "make_navigator",
    "quicklook",
    "summary_table",
]


def computed(signal, show_progressbar=False):
    """Compute a lazy signal and *return* it.

    ``LazySignal.compute()`` works in place and returns ``None``, so
    ``m = s.sum(...).compute()`` silently gives ``None``.  Use::

        m = gi.computed(hl.isig[840.:900.].sum(axis="Energy loss"))
    """
    if hasattr(signal, "compute") and getattr(signal, "_lazy", False):
        signal.compute(show_progressbar=show_progressbar)
    return signal


def _nav_shape(signal):
    """The signal's navigation shape in *array* order."""
    return signal.data.shape[: signal.axes_manager.navigation_dimension]


def make_navigator(signal, image=None, energy_window=None, show_progressbar=False,
                   *, haadf=None):
    """Build a cheap, already-computed navigator for a lazy in-situ signal.

    Pass exactly one source:

    ``image``
        The simultaneous image stream -- HAADF, ADF, DF, whatever the session
        used; :func:`image_streams` finds it without you naming a detector.
        It is recorded at an integer multiple of the SI sampling
        (``SI.Scale``, 4x here: 472x444 at 0.0318 nm against 118x111 at
        0.1271 nm), so it rebins exactly onto the navigation grid.  Only
        ~40 MB, and it is the most faithful navigator you can have -- it is
        the real simultaneous image.  (``haadf=`` still works as an alias.)
    ``energy_window``
        ``(start_eV, stop_eV)``.  Sums that window per probe position.  Reads
        only the chunks covering the window, so a narrow one is cheap, and the
        result is an elemental map rather than a slab of arbitrary channels.

    Returns a computed signal whose navigation axes are copied from ``signal``,
    ready for ``signal.plot(navigator=nav)`` or ``signal.navigator = nav``.
    """
    image = image if image is not None else haadf      # haadf= is the old name
    if (image is None) == (energy_window is None):
        raise ValueError("pass exactly one of image= or energy_window=")

    target = _nav_shape(signal)

    if image is not None:
        source = image.data.shape
        if len(source) != len(target):
            raise ValueError(
                f"the image has {len(source)} dimensions but the signal has "
                f"{len(target)} navigation dimensions ({source} vs {target})"
            )
        ratios = []
        for got, want in zip(source, target):
            ratio, remainder = divmod(got, want)
            if remainder or ratio < 1:
                raise ValueError(
                    f"image shape {source} is not an integer multiple of the "
                    f"navigation shape {target}; cannot rebin onto the SI grid"
                )
            ratios.append(ratio)
        navigator = (image.rebin(scale=tuple(ratios))
                     if any(r > 1 for r in ratios) else image.deepcopy())
        navigator = computed(navigator, show_progressbar)
        label = image.metadata.General.title or "image"
    else:
        start, stop = energy_window
        window = signal.isig[float(start) : float(stop)]
        navigator = computed(
            window.sum(axis=signal.axes_manager.signal_axes[0].name), show_progressbar
        )
        label = f"{start:g}-{stop:g} eV"

    if navigator.data.shape != tuple(target):
        raise ValueError(
            f"navigator shape {navigator.data.shape} does not match the "
            f"navigation shape {tuple(target)}"
        )

    # Copy the navigation calibration across so the sliders read the same.
    nav_axes = [a for a in signal.axes_manager._axes][: len(target)]
    for axis, reference in zip(navigator.axes_manager._axes, nav_axes):
        axis.name = reference.name
        axis.units = reference.units
        if hasattr(axis, "scale") and hasattr(reference, "scale"):
            axis.scale = reference.scale
            axis.offset = reference.offset
    navigator.metadata.General.title = f"{label} navigator"
    return navigator


def summary_table(folder):
    """List an acquisition folder as a pandas DataFrame, reading no bulk data."""
    import pandas as pd
    from pathlib import Path

    rows = []
    for dm4 in sorted(Path(folder).glob("*.dm4")):
        info = read_raw_format_info(dm4)
        if info is None:
            rows.append(
                {
                    "name": _short_name(dm4.stem),
                    "frames": 1,
                    "frame shape": "-",
                    "dtype": "-",
                    "raw files": 0,
                    "size (GB)": round(dm4.stat().st_size / 1e9, 3),
                    "dt (s)": None,
                    "energy (eV)": None,
                    "pixel (nm)": None,
                }
            )
            continue
        energy = next((a for a in info.frame_axes if a["units"] == "eV"), None)
        spatial = next((a for a in info.frame_axes if a["units"] == "nm"), None)
        rows.append(
            {
                "name": _short_name(dm4.stem),
                "frames": info.n_frames_declared,
                "frame shape": " x ".join(str(s) for s in info.frame_shape),
                "dtype": info.dtype.name,
                "raw files": len(info.raw_paths),
                "size (GB)": round(
                    info.frame_bytes * info.n_frames_declared / 1e9, 3
                ),
                "dt (s)": (
                    round(info.frame_interval, 3) if info.frame_interval else None
                ),
                "energy (eV)": (
                    f"{energy['offset']:.0f} .. "
                    f"{energy['offset'] + (energy['size'] - 1) * energy['scale']:.0f}"
                    f" @ {energy['scale']:.2f}"
                    if energy
                    else None
                ),
                "pixel (nm)": round(spatial["scale"], 5) if spatial else None,
            }
        )
    return pd.DataFrame(rows).set_index("name")


def _plot_spectrum(axis, energy, counts, title):
    """Plot a summed spectrum, choosing log scale only when it helps.

    Dark-corrected EELS has zero and negative channels, which a log axis draws
    as spikes to the bottom of the plot, so non-positive values are masked
    rather than clipped.
    """
    positive = counts[counts > 0]
    use_log = positive.size > 0 and positive.max() / positive.min() > 1e3
    axis.plot(
        energy,
        np.where(counts > 0, counts, np.nan) if use_log else counts,
        lw=0.8,
        color="C0",
    )
    if use_log:
        axis.set_yscale("log")
    axis.set_xlabel("energy loss (eV)")
    axis.set_ylabel("counts (summed)" + (" [log]" if use_log else ""))
    axis.set_title(title, fontsize=9)


#: Image-stream names worth preferring, best first.  Which detector was used
#: varies between sessions -- HAADF on one, ADF or plain DF on another -- so
#: nothing downstream should test for one name.
IMAGE_PREFERENCE = ("HAADF", "ADF", "DF", "BF", "ABF", "SE", "IMAGE")


def classify_signals(signals):
    """Sort a loaded dataset into the kinds of stream a figure can show.

    Returns ``(spectra, series, stills, other)`` dicts.  The distinction that
    matters is **navigation** dimension, not signal dimension: an EDS or EELS
    spectrum image stored in its native ``(time, E | x, y)`` layout has a 2D
    *signal* just like an image series does, but two navigation axes, so
    ``data[0]`` is a stack rather than a picture.  Classifying on signal
    dimension alone is what made ``quicklook`` raise
    ``TypeError: Invalid shape ... for image data`` on such a dataset.

    * ``spectra`` -- one signal axis (EELS and anything else 1D)
    * ``series``  -- 2D signal, one navigation axis: a frame series
    * ``stills``  -- 2D signal, no navigation axes: a survey image
    * ``other``   -- anything else, which no single panel can represent
    """
    spectra, series, stills, other = {}, {}, {}, {}
    for key, signal in signals.items():
        nav = signal.axes_manager.navigation_dimension
        sig = signal.axes_manager.signal_dimension
        if sig == 1:
            spectra[key] = signal
        elif sig == 2 and nav == 1:
            series[key] = signal
        elif sig == 2 and nav == 0:
            stills[key] = signal
        else:
            other[key] = signal
    return spectra, series, stills, other


def image_streams(signals, prefer=IMAGE_PREFERENCE):
    """Keys of the image *series* in a dataset, most detector-like first.

    Use this instead of looking for ``'HAADF'`` in the key: the same workflow
    sees HAADF one session and ADF the next, and a dataset can also carry
    derived image series (a correlation map, a peak shift) that are the same
    shape but are not detector images.  Names matching ``prefer`` come first,
    in that order; everything else keeps its original order behind them.
    """
    _, series, _, _ = classify_signals(signals)

    def rank(key):
        upper = key.upper()
        for position, name in enumerate(prefer):
            if name in upper:
                return position
        return len(prefer)

    return sorted(series, key=lambda k: (rank(k), list(series).index(k)))


def frame_range(n_frames, start=0, stop=0):
    """Resolve a user's ``start``/``stop`` into a concrete list of frames.

    ``stop=0`` means "to the end", so a batch parameter file can leave both at
    zero and get the whole series.  Negative values count from the end.  The
    point of this is beam damage: when the last frames of a series are
    destroyed, everything that sums or aligns should use the same subset, and
    the subset should be recorded, not applied ad hoc in one cell.
    """
    start = int(start)
    stop = int(stop)
    if start < 0:
        start += n_frames
    if stop <= 0:
        stop += n_frames
    start = max(0, min(start, n_frames))
    stop = max(0, min(stop, n_frames))
    if stop <= start:
        raise ValueError(
            f"empty frame range: start={start}, stop={stop} of {n_frames} frames"
        )
    return list(range(start, stop))


def save_frame_gif(
    signal,
    path,
    *,
    frames=None,
    fps=5,
    label=True,
    times=None,
    cmap="gray",
    percentiles=(1, 99),
    scale_per_frame=False,
    width=400,
    dpi=100,
):
    """Write an animated GIF of an image series, with the frame number burnt in.

    The point of the label is that an animation with nothing written on it
    cannot be referred to: once you spot the frame where the specimen changes,
    you need its number to set ``start``/``stop`` or to go back to it.

    Parameters
    ----------
    signal : image series, ``(frames, y, x)``
    path : str or Path
        Written as GIF; the suffix is forced to ``.gif``.
    frames : sequence of int, optional
        Which frames to include -- pass :func:`frame_range` output to show only
        the ones you are keeping.
    label : bool
        Burn ``frame N`` (and the time, if available) into the corner.
    times : array, optional
        Seconds per frame, for the label.  Taken from a ``Time`` axis or from
        ``metadata.InSitu.frame_times_s`` when not given.
    percentiles : (float, float)
        Contrast limits, as percentiles of the **whole series** so the
        brightness does not flicker frame to frame.  ``scale_per_frame=True``
        rescales each frame instead, which exaggerates drift in the detector
        gain but shows faint detail in a dimming series.
    width : int
        Output width in pixels; the height follows the aspect ratio.

    Returns the path written.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from pathlib import Path

    path = Path(path).with_suffix(".gif")
    data = signal.data if hasattr(signal, "data") else signal
    if hasattr(data, "compute"):
        data = data.compute()
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(
            f"need an image series shaped (frames, y, x), got {data.shape}"
        )

    order = list(range(data.shape[0])) if frames is None else [int(f) for f in frames]
    if not order:
        raise ValueError("no frames selected")

    if times is None and hasattr(signal, "axes_manager"):
        names = [a.name for a in signal.axes_manager.navigation_axes]
        time_name = next((n for n in names if n.lower().startswith("time")), None)
        if time_name is not None:
            times = signal.axes_manager[time_name].axis
        else:
            times = signal.metadata.get_item("InSitu.frame_times_s")
    times = None if times is None else np.asarray(times, dtype=float)

    block = data[order]
    finite = block[np.isfinite(block)]
    low, high = (np.percentile(finite, percentiles) if finite.size
                 else (0.0, 1.0))
    if high <= low:
        high = low + 1.0

    height = max(1, int(round(width * data.shape[1] / data.shape[2])))
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_axis_off()
    picture = axis.imshow(block[0], cmap=cmap, vmin=low, vmax=high,
                          interpolation="nearest")
    caption = axis.text(
        0.02, 0.975, "", transform=axis.transAxes, ha="left", va="top",
        fontsize=9, color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
    )

    def draw(index):
        frame = block[index]
        picture.set_data(frame)
        if scale_per_frame:
            good = frame[np.isfinite(frame)]
            if good.size:
                picture.set_clim(*np.percentile(good, percentiles))
        if label:
            number = order[index]
            text = f"frame {number}"
            if times is not None and number < times.size:
                text += f"   {times[number]:.0f} s"
            caption.set_text(text)
        return picture, caption

    writer = animation.PillowWriter(fps=float(fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, str(path), dpi=dpi):
        for index in range(len(order)):
            draw(index)
            writer.grab_frame()
    plt.close(figure)
    _logger.info(
        "wrote %s: %d frames at %g fps, %dx%d", path.name, len(order), fps, width, height
    )
    return path


def quicklook(signals, figsize=None, show_progressbar=False, *,
              images=None, spectra=None, max_image_panels=6):
    """Draw a one-figure overview of a loaded in-situ dataset.

    Takes the dict from :func:`~gatan_insitu.load_insitu_dataset`.  Top row:
    the survey image and the first/last frame of each image series, so any
    drift or specimen change is immediately visible.  Bottom row: the summed
    spectrum of each spectral stream plus the intensity-versus-time trend.

    Each stream is read exactly once -- ``sum`` over ``y`` and ``x`` gives both
    the summed spectrum (sum over time) and the time trend (sum over energy)
    from the same pass.

    Parameters
    ----------
    images, spectra : sequence of str, optional
        Which streams to show.  By default the image series are chosen by
        :func:`image_streams` (so the detector image comes first whether it is
        called HAADF, ADF or DF) and every 1D-signal stream is plotted.
    max_image_panels : int
        Cap on the top row, so a dataset carrying several derived image series
        does not produce a figure a metre wide.

    Streams that no single panel can represent -- a spectrum image in its
    native layout, for instance -- are skipped and named in the figure, rather
    than raising.
    """
    import matplotlib.pyplot as plt

    found_spectra, series, stills, other = classify_signals(signals)

    if spectra is None:
        eels = found_spectra
    else:
        eels = {k: signals[k] for k in spectra if k in signals}
    chosen = image_streams(signals) if images is None else list(images)
    chosen = [k for k in chosen if k in series]

    # Top row: every still, plus first and last frame of each image series.
    panels = [(key, np.asarray(signal.data)) for key, signal in stills.items()]
    for key in chosen:
        signal = series[key]
        last = signal.data.shape[0] - 1
        panels.append((f"{key}\nframe 0", np.asarray(signal.data[0])))
        if last:
            panels.append((f"{key}\nframe {last}", np.asarray(signal.data[last])))
    if len(panels) > max_image_panels:
        _logger.info(
            "showing %d of %d image panels; pass images=[...] to choose",
            max_image_panels, len(panels),
        )
        panels = panels[:max_image_panels]

    n_cols = max(len(panels), len(eels) + 1, 1)
    figsize = figsize or (min(2.6 * n_cols, 20), 7.5)
    figure, axes = plt.subplots(2, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:  # n_cols == 1 collapses the array
        axes = axes.reshape(2, -1)

    for column in range(n_cols):
        axis = axes[0, column]
        if column < len(panels):
            title, image = panels[column]
            axis.imshow(image, cmap="gray")
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            axis.axis("off")

    # Bottom row: one spectrum per spectral stream, then the shared time trend.
    trends = {}
    shown = 0
    for key, signal in eels.items():
        if shown >= n_cols - 1:
            break
        # Sum over every navigation axis except time, whatever they are called:
        # a stream need not have 'y' and 'x' (a line scan, an already-summed
        # series), and hard-coding them is how this breaks on the next dataset.
        nav_names = [a.name for a in signal.axes_manager.navigation_axes]
        time_name = next((n for n in nav_names if n.lower().startswith("time")), None)
        reduce_over = [n for n in nav_names if n != time_name]
        per_frame = signal
        if reduce_over:
            per_frame = computed(
                signal.sum(axis=reduce_over), show_progressbar=show_progressbar
            )
        else:
            per_frame = computed(signal.deepcopy(), show_progressbar=show_progressbar)
        values = np.atleast_2d(np.asarray(per_frame.data))
        energy = per_frame.axes_manager[signal.axes_manager.signal_axes[0].name].axis
        _plot_spectrum(axes[1, shown], energy, values.sum(axis=0),
                       f"{key}: summed spectrum")
        if time_name is not None and values.shape[0] > 1:
            trends[key] = (per_frame.axes_manager[time_name].axis, values.sum(axis=1))
        shown += 1

    trend_axis = axes[1, shown] if shown < n_cols else None
    if trend_axis is not None and trends:
        for key, (time, total) in trends.items():
            trend_axis.plot(time, total / total.mean(), ".-", ms=4, lw=0.9, label=key)
        trend_axis.set_xlabel("time (s)")
        trend_axis.set_ylabel("total counts / mean")
        trend_axis.set_title("intensity vs time", fontsize=9)
        trend_axis.legend(fontsize=7)
        shown += 1
    for column in range(shown, n_cols):
        axes[1, column].axis("off")

    if other:
        names = ", ".join(other)
        _logger.info("not shown (no single panel represents them): %s", names)
        figure.text(0.5, 0.005, f"not shown: {names}", ha="center", fontsize=7,
                    color="0.4")

    return figure
