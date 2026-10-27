"""Helpers for working with multiframe in-situ records in a Jupyter notebook.

Three things that bite in a notebook and are fixed here:

* **Navigators.** ``plot()`` on a lazy signal builds its navigator by summing
  over a *single* chunk, so the image you steer with shows ~315 of 3456 energy
  channels rather than the whole spectrum, and it takes a few seconds.
  :func:`make_navigator` gives a real one instead -- ideally the simultaneously
  recorded HAADF stream, which rebins exactly onto the SI navigation grid.
* **``compute()`` returns ``None``.** It works in place, so the obvious
  one-liner throws away the result.  :func:`computed` returns the signal.
* **First look.** :func:`quicklook` draws survey, HAADF, summed spectra and the
  intensity-versus-time trend in one pass over the data, and
  :func:`summary_table` lists a folder as a DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np

from .dm_tags import read_raw_format_info
from .loader import _short_name

_logger = logging.getLogger(__name__)

__all__ = ["computed", "make_navigator", "quicklook", "summary_table"]


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


def make_navigator(signal, haadf=None, energy_window=None, show_progressbar=False):
    """Build a cheap, already-computed navigator for a lazy in-situ signal.

    Pass exactly one source:

    ``haadf``
        The HAADF stream from the same acquisition.  It is recorded at an
        integer multiple of the SI sampling (``SI.Scale``, 4x here: 472x444 at
        0.0318 nm against 118x111 at 0.1271 nm), so it rebins exactly onto the
        navigation grid.  Only ~40 MB, and it is the most faithful navigator
        you can have -- it is the real simultaneous image.
    ``energy_window``
        ``(start_eV, stop_eV)``.  Sums that window per probe position.  Reads
        only the chunks covering the window, so a narrow one is cheap, and the
        result is an elemental map rather than a slab of arbitrary channels.

    Returns a computed signal whose navigation axes are copied from ``signal``,
    ready for ``signal.plot(navigator=nav)`` or ``signal.navigator = nav``.
    """
    if (haadf is None) == (energy_window is None):
        raise ValueError("pass exactly one of haadf= or energy_window=")

    target = _nav_shape(signal)

    if haadf is not None:
        source = haadf.data.shape
        if len(source) != len(target):
            raise ValueError(
                f"HAADF has {len(source)} dimensions but the signal has "
                f"{len(target)} navigation dimensions ({source} vs {target})"
            )
        ratios = []
        for got, want in zip(source, target):
            ratio, remainder = divmod(got, want)
            if remainder or ratio < 1:
                raise ValueError(
                    f"HAADF shape {source} is not an integer multiple of the "
                    f"navigation shape {target}; cannot rebin onto the SI grid"
                )
            ratios.append(ratio)
        navigator = haadf.rebin(scale=tuple(ratios)) if any(r > 1 for r in ratios) else haadf.deepcopy()
        navigator = computed(navigator, show_progressbar)
        label = "HAADF"
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


def quicklook(signals, figsize=(14, 7.5), show_progressbar=False):
    """Draw a one-figure overview of a loaded in-situ dataset.

    Takes the dict from :func:`~gatan_insitu.load_insitu_dataset`.  Top row:
    the survey image and the first/last frame of each image stream, so any
    drift or specimen change is immediately visible.  Bottom row: the summed
    spectrum of each EELS stream plus the intensity-versus-time trend.

    Each 9 GB stream is read exactly once -- ``sum`` over ``y`` and ``x`` gives
    both the summed spectrum (sum over time) and the time trend (sum over
    energy) from the same pass.
    """
    import matplotlib.pyplot as plt

    eels, images = {}, {}
    for key, signal in signals.items():
        if signal.metadata.get_item("Signal.signal_type", "") == "EELS":
            eels[key] = signal
        elif signal.axes_manager.signal_dimension == 2:
            images[key] = signal

    # Top row: every still, plus first and last frame of every image stream.
    panels = []
    for key, signal in images.items():
        if signal.axes_manager.navigation_dimension:
            last = signal.data.shape[0] - 1
            panels.append((f"{key}\nframe 0", np.asarray(signal.data[0])))
            panels.append((f"{key}\nframe {last}", np.asarray(signal.data[last])))
        else:
            panels.append((key, np.asarray(signal.data)))

    n_cols = max(len(panels), len(eels) + 1, 1)
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

    # Bottom row: one spectrum per EELS stream, then the shared time trend.
    trends = {}
    for column, (key, signal) in enumerate(eels.items()):
        per_frame = computed(
            signal.sum(axis=["y", "x"]), show_progressbar=show_progressbar
        )
        energy = per_frame.axes_manager[signal.axes_manager.signal_axes[0].name].axis
        _plot_spectrum(
            axes[1, column],
            energy,
            per_frame.data.sum(axis=0),
            f"{key}: summed spectrum",
        )
        trends[key] = (per_frame.axes_manager["Time"].axis, per_frame.data.sum(axis=1))

    trend_axis = axes[1, len(eels)] if len(eels) < n_cols else None
    if trend_axis is not None:
        for key, (time, total) in trends.items():
            trend_axis.plot(time, total / total.mean(), ".-", ms=4, lw=0.9, label=key)
        trend_axis.set_xlabel("time (s)")
        trend_axis.set_ylabel("total counts / mean")
        trend_axis.set_title("intensity vs time", fontsize=9)
        if trends:
            trend_axis.legend(fontsize=7)
    for column in range(len(eels) + 1, n_cols):
        axes[1, column].axis("off")

    return figure
