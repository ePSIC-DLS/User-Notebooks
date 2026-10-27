"""Frame-to-frame (spatial) drift alignment of a multiframe in-situ record.

The specimen drifts between frames, so the same probe position in two frames
is not the same place on the specimen.  The drift is measured on an image
stream -- the simultaneous HAADF, ideally, which is sharp and recorded at an
integer multiple of the SI sampling -- and then applied to *every* stream of
the acquisition, so the HAADF, the high loss and the low loss stay registered
with each other.

Two design choices, both deliberate:

* **Integer shifts on the coarsest grid.** The drift is measured to sub-pixel
  precision on the fine HAADF, then rounded to whole pixels of the coarsest
  stream (the SI).  Every stream is then *cropped*, never interpolated: a
  spectrum is never resampled, so its counts and noise are untouched, and
  because the HAADF is cropped by exactly ``ratio`` times the SI shift, the
  aligned HAADF still rebins exactly onto the aligned SI.  The cost is up to
  half an SI pixel of residual misregistration, which is reported.
* **Crop, don't pad.** The aligned series keeps only the field of view common
  to all frames, so there are no empty borders to mask.  It stays lazy: each
  frame is just a different slice of the original array.

    hl_al, ll_al, haadf_al, report = gi.align_frames(haadf, hl, ll)
"""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger(__name__)

__all__ = ["frame_drift", "align_frames"]


def _frames(image):
    """``(frames, y, x)`` float array from a Signal2D series or an array."""
    data = getattr(image, "data", image)
    if hasattr(data, "compute"):
        data = data.compute()
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected a (frames, y, x) image series, got shape {data.shape}")
    return data


def _prepare(frame, smooth):
    from scipy.ndimage import gaussian_filter
    from skimage.filters import window

    if smooth:
        frame = gaussian_filter(frame, smooth)
    frame = frame - frame.mean()
    # A Hann window stops the image edges, which do not match from frame to
    # frame, from dominating the correlation.
    return frame * window("hann", frame.shape)


def frame_drift(image, *, reference="previous", upsample=10, smooth=1.0):
    """Measure the specimen drift of every frame relative to frame 0.

    Parameters
    ----------
    image : Signal2D series or ``(frames, y, x)`` array
        The image stream to measure on, usually the HAADF.
    reference : {'previous', 'first'}
        ``'previous'`` (default) correlates each frame with the one before and
        accumulates the steps.  It copes with a specimen that changes through
        the series -- neighbouring frames always look alike -- at the price of
        a random-walk error that grows slowly with the number of frames.
        ``'first'`` correlates every frame with frame 0: no accumulation, but it
        fails once the specimen has changed too much to match frame 0.
    upsample : int
        Sub-pixel precision of the phase correlation, 1/upsample pixel.
    smooth : float
        Gaussian sigma (pixels) applied before correlating, to suppress noise.

    Returns
    -------
    ``(frames, 2)`` array of ``(dy, dx)`` in pixels of ``image``: a feature at
    ``p`` in frame 0 is at ``p + drift[k]`` in frame ``k``.  ``drift[0]`` is 0.
    """
    from skimage.registration import phase_cross_correlation

    if reference not in ("previous", "first"):
        raise ValueError("reference must be 'previous' or 'first'")
    data = _frames(image)
    prepared = [_prepare(frame, smooth) for frame in data]

    drift = np.zeros((len(prepared), 2))
    for k in range(1, len(prepared)):
        base = prepared[k - 1] if reference == "previous" else prepared[0]
        # phase_cross_correlation returns the shift that *registers* the moving
        # image onto the reference, i.e. minus the displacement of its content.
        shift, _, _ = phase_cross_correlation(
            base, prepared[k], upsample_factor=upsample, normalization=None
        )
        step = -np.asarray(shift, dtype=float)
        drift[k] = (drift[k - 1] + step) if reference == "previous" else step
    return drift


def _spatial_shape(signal):
    am = signal.axes_manager
    return am["y"].size, am["x"].size


def _crop_series(signal, shifts):
    """Crop every frame of ``signal`` to the common field, following ``shifts``.

    ``shifts`` are integer ``(dy, dx)`` per frame in the signal's own pixels.
    Returns a new signal (lazy if the input is) with the frame-0 window's
    calibration, so its spatial offsets say where the kept field sits.
    """
    am = signal.axes_manager
    if am["Time"].index_in_array != 0:
        raise ValueError(f"{signal}: expected Time as the first array axis")
    iy, ix = am["y"].index_in_array, am["x"].index_in_array
    ny, nx = signal.data.shape[iy], signal.data.shape[ix]

    lo, hi = shifts.min(axis=0), shifts.max(axis=0)
    size_y, size_x = ny - (hi[0] - lo[0]), nx - (hi[1] - lo[1])
    if size_y < 1 or size_x < 1:
        raise ValueError(f"drift ({hi - lo} px) exceeds the field of view ({ny}, {nx})")
    starts = shifts - lo  # window start in each frame

    if signal._lazy:
        import dask.array as da

        stack = da.stack
    else:
        stack = np.stack
    frames = []
    for k, (sy, sx) in enumerate(starts):
        index = [slice(None)] * signal.data.ndim
        index[0] = k
        index[iy] = slice(int(sy), int(sy) + int(size_y))
        index[ix] = slice(int(sx), int(sx) + int(size_x))
        frames.append(signal.data[tuple(index)])
    data = stack(frames, axis=0)

    # Slice through HyperSpy with frame 0's window to get correctly sized and
    # calibrated axes, then swap in the per-frame data (same shape).
    y0, x0 = int(starts[0, 0]), int(starts[0, 1])
    ys, xs = slice(y0, y0 + int(size_y)), slice(x0, x0 + int(size_x))
    template = signal
    if am["x"].navigate:
        template = template.inav[xs, ys]
    else:
        template = template.isig[xs, ys]
    template.data = data
    return template


def align_frames(
    reference,
    *also_align,
    drift=None,
    frame_reference="previous",
    upsample=10,
    smooth=1.0,
):
    """Align every frame of a multiframe record for specimen drift.

    The drift is measured on ``reference`` (the HAADF series), rounded to whole
    pixels of the coarsest stream, and every stream is cropped to the field
    common to all frames.  Nothing is interpolated and everything stays lazy.

    Parameters
    ----------
    reference : Signal2D series
        Image stream to measure the drift on, with a ``Time`` navigation axis.
    *also_align : signals
        Other streams of the same acquisition (the EELS high and low loss).
        Their ``y``/``x`` grid must divide the reference grid by an integer --
        the HAADF is recorded at ``SI.Scale`` (4x here) the SI sampling.
    drift : ``(frames, 2)`` array, optional
        Drift in reference pixels, from :func:`frame_drift`, to skip measuring.
    frame_reference, upsample, smooth :
        Passed to :func:`frame_drift`.

    Returns
    -------
    ``(reference, *also_align, report)`` -- new aligned signals, in the order
    given, and a report with the measured drift, the applied shifts, the
    residual misregistration and how much of the field was kept.
    """
    streams = (reference,) + tuple(also_align)
    n_frames = reference.axes_manager["Time"].size
    for signal in also_align:
        if signal.axes_manager["Time"].size != n_frames:
            raise ValueError(f"{signal} has {signal.axes_manager['Time'].size} frames, "
                             f"the reference has {n_frames}")

    ref_shape = np.array(_spatial_shape(reference))
    ratios = []
    for signal in streams:
        ratio, remainder = np.divmod(ref_shape, _spatial_shape(signal))
        if np.any(remainder) or ratio[0] != ratio[1] or ratio[0] < 1:
            raise ValueError(
                f"{signal}: grid {_spatial_shape(signal)} is not an integer fraction "
                f"of the reference grid {tuple(ref_shape)}"
            )
        ratios.append(int(ratio[0]))
    coarse = max(ratios)  # reference pixels per coarsest-grid pixel

    if drift is None:
        drift = frame_drift(reference, reference=frame_reference, upsample=upsample,
                            smooth=smooth)
    drift = np.asarray(drift, dtype=float)

    # Whole pixels of the coarsest grid, so every stream gets an integer shift.
    coarse_shifts = np.rint(drift / coarse).astype(int)
    residual = drift - coarse_shifts * coarse  # in reference pixels

    aligned = []
    for signal, ratio in zip(streams, ratios):
        shifts = coarse_shifts * (coarse // ratio)
        out = _crop_series(signal, shifts)
        out.metadata.set_item("Processing.frame_alignment.shifts_px", shifts)
        out.metadata.set_item("Processing.frame_alignment.measured_on",
                              reference.metadata.General.title)
        aligned.append(out)

    ref_scale = reference.axes_manager["x"].scale
    kept = np.prod(_spatial_shape(aligned[0])) / np.prod(ref_shape)
    report = {
        "drift_px": drift,
        "drift_nm": drift * ref_scale,
        "applied_shifts_coarse_px": coarse_shifts,
        "coarse_factor": coarse,
        "residual_px": residual,
        "residual_rms_nm": float(np.sqrt(np.mean(residual ** 2)) * ref_scale),
        "total_drift_nm": float(np.hypot(*(drift[-1] - drift[0])) * ref_scale),
        "field_kept": float(kept),
        "frame_reference": frame_reference,
    }
    _logger.info(
        "frame drift %.2f nm over %d frames; kept %.0f%% of the field, "
        "residual %.3f nm rms", report["total_drift_nm"], n_frames, 100 * kept,
        report["residual_rms_nm"],
    )
    return tuple(aligned) + (report,)
