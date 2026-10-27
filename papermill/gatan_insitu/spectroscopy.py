"""Edge intensities and white-line ratios, frame by frame, for an in-situ series.

Built for the per-region spectra that :func:`~gatan_insitu.masked_spectra`
returns, so an edge can be followed through the stack separately in each
region.

Two measurements, both the standard recipes:

``edge_intensity``
    Fit a power law over a pre-edge window, subtract it, integrate the edge.
    Used here for O K.  Divide two of these to get a composition-like
    ratio that cancels thickness.

``white_line_ratio``
    The Pearson method for L3/L2.  After the power law comes a **double
    arctan step**, representing transitions to continuum states, with the
    2:1 height ratio set by the 4:2 degeneracy of the L3 and L2 initial
    states.  Skipping that step is the usual way to get a wrong L3/L2:
    the continuum sits under both white lines and under L2 it is a much
    larger fraction of the signal, so leaving it in drags the ratio down
    (1.22 against 3.26 on this data).

Everything carries an uncertainty estimated from the scatter left in the
pre-edge window after background subtraction, and :func:`trend` says whether a
series actually changes across the stack or is just noise.  On a weak edge the
per-frame numbers are often dominated by noise -- :func:`bin_frames` trades
time resolution for signal when that happens.
"""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger(__name__)

# numpy renamed trapz -> trapezoid in 2.0; this env has 1.26.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "align_to_zlp",
    "bin_frames",
    "edge_intensity",
    "edge_table",
    "fit_white_lines",
    "fixed_pattern",
    "hyperspy_white_lines",
    "power_law_background",
    "remove_fixed_pattern",
    "rolling_frames",
    "trend",
    "white_line_ratio",
    "zlp_position",
]


def _unpack(spectra):
    """Normalise the input into ``[(label, energy, data (positions, channels))]``.

    Navigation dimensions are flattened, so these routines work on a series of
    region spectra ``(frames, E)`` and equally on a full spectrum image
    ``(frames, y, x, E)``; results then run over the flattened positions.
    """
    if hasattr(spectra, "axes_manager"):
        spectra = {spectra.metadata.General.title or "signal": spectra}
    items = []
    for label, signal in spectra.items():
        if not hasattr(signal, "axes_manager"):
            raise TypeError(f"{label!r} is not a HyperSpy signal")
        energy = np.asarray(signal.axes_manager.signal_axes[0].axis, dtype=float)
        values = signal.data
        if hasattr(values, "compute"):
            values = values.compute()
        data = np.atleast_2d(np.asarray(values, dtype=float))
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])
        items.append((label, energy, data))
    return items


def _window(energy, limits):
    mask = (energy >= limits[0]) & (energy <= limits[1])
    if mask.sum() < 3:
        raise ValueError(
            f"window {limits} selects {mask.sum()} channels of the "
            f"{energy[0]:.0f}-{energy[-1]:.0f} eV axis"
        )
    return mask


def power_law_background(energy, counts, fit_window):
    """Fit ``A * E**-r`` over ``fit_window`` and evaluate it everywhere.

    Fitted as a straight line in log-log space, which needs no starting guess
    and cannot diverge -- unlike iterative power-law fitting, which on
    dark-corrected data containing zeros and negatives throws overflow and
    invalid-value warnings and sometimes fails outright.

    Returns an array of NaN if the window holds too few positive channels.
    """
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if energy.min() <= 0:
        raise ValueError("a power-law background needs a positive energy axis")
    mask = _window(energy, fit_window) & (counts > 0) & np.isfinite(counts)
    if mask.sum() < 5:
        return np.full_like(counts, np.nan)
    slope, intercept = np.polyfit(np.log(energy[mask]), np.log(counts[mask]), 1)
    return np.exp(intercept) * energy**slope


def _noise(energy, residual, fit_window):
    """Per-channel noise, from the scatter left where the fit was made."""
    mask = _window(energy, fit_window) & np.isfinite(residual)
    return float(np.std(residual[mask])) if mask.sum() > 2 else np.nan


def _integrate(energy, values, mask):
    return float(_trapezoid(values[mask], energy[mask]))


def edge_intensity(spectra, window, fit_window, *, with_background=False):
    """Background-subtracted integral of an edge, for every frame.

    Parameters
    ----------
    window : (float, float)
        Integration range, in eV, starting at the edge onset.
    fit_window : (float, float)
        Pre-edge range the power law is fitted over.  It must contain no edge.
    with_background : bool
        Also return the integral of the fitted background over ``window`` -- a
        useful thickness proxy to normalise against.

    Returns
    -------
    dict of ``{label: {'intensity', 'error', 'background'}}``, each an array
    over frames.  ``error`` is propagated from the pre-edge scatter.
    """
    out = {}
    for label, energy, data in _unpack(spectra):
        inside = _window(energy, window)
        width = float(energy[inside][-1] - energy[inside][0])
        step = float(np.median(np.diff(energy)))
        values, errors, backgrounds = [], [], []
        for frame in data:
            background = power_law_background(energy, frame, fit_window)
            subtracted = frame - background
            values.append(_integrate(energy, subtracted, inside))
            sigma = _noise(energy, subtracted, fit_window)
            errors.append(sigma * step * np.sqrt(inside.sum()) if np.isfinite(sigma) else np.nan)
            backgrounds.append(_integrate(energy, background, inside))
        result = {
            "intensity": np.array(values),
            "error": np.array(errors),
            "window_eV": width,
        }
        if with_background:
            result["background"] = np.array(backgrounds)
        else:
            result["background"] = np.array(backgrounds)
        out[label] = result
        failures = int(np.sum(~np.isfinite(result["intensity"])))
        if failures:
            _logger.warning("%s: background fit failed on %d frames", label, failures)
    return out


def _double_step(energy, onsets, width, ratio):
    """Normalised double arctan step: 0 below L3, 1 well above L2."""
    low = np.arctan((energy - onsets[0]) / width) / np.pi + 0.5
    high = np.arctan((energy - onsets[1]) / width) / np.pi + 0.5
    return (ratio * low + high) / (ratio + 1.0)


def white_line_ratio(
    spectra,
    *,
    l3=(851.0, 859.0),
    l2=(867.4, 877.4),
    fit_window=(800.0, 845.0),
    continuum=(895.0, 925.0),
    onsets=(855.0, 872.4),
    step_width=1.0,
    step_ratio=2.0,
    subtract_step=True,
    min_snr=3.0,
):
    """L3/L2 white-line ratio per frame (Pearson method).

    Defaults are for Ni L2,3 and were read off this data: the white lines sit
    at 855.0 and 872.4 eV (17.5 eV apart) and the continuum settles by ~895 eV.
    Check them against your own spectrum before using them elsewhere.

    **Check ``l2_snr`` in the result before believing the ratio.** The windows
    must be tight around the peaks: these white lines are only a few eV wide,
    and a window wide enough to include the dip between them drives the L2
    integral towards zero and makes the ratio explode.

    Parameters
    ----------
    l3, l2 : (float, float)
        Integration windows for the two white lines.  Keep them narrow.
    fit_window : (float, float)
        Pre-edge range for the power-law background.
    continuum : (float, float)
        A range well above L2 where only the continuum step remains; its
        height is matched here.
    onsets : (float, float)
        Where the two steps rise -- the white-line positions.
    step_width : float
        Arctan width in eV, of order the energy resolution plus lifetime
        broadening.
    step_ratio : float
        Height ratio of the L3 to the L2 step, 2:1 from the 4:2 degeneracy.
    subtract_step : bool
        Set ``False`` to get the raw windowed ratio instead. That is **not**
        a white-line ratio and is not comparable with published values; it is
        here only so the difference can be seen.
    min_snr : float
        Warn when the median L2 peak height falls below this many times the
        baseline noise. A ratio measured on an undetected L2 is meaningless,
        and that is easy to miss because it still returns a number.

    Returns
    -------
    dict of ``{label: {'l3', 'l2', 'ratio', 'error', 'step_height',
    'l3_snr', 'l2_snr'}}``.  The SNRs are peak height over baseline noise,
    per frame.
    """
    out = {}
    for label, energy, data in _unpack(spectra):
        w3, w2 = _window(energy, l3), _window(energy, l2)
        wc = _window(energy, continuum)
        shape = _double_step(energy, onsets, step_width, step_ratio)
        step_size = float(np.median(np.diff(energy)))

        l3_values, l2_values, errors, heights = [], [], [], []
        snr3, snr2 = [], []
        for frame in data:
            background = power_law_background(energy, frame, fit_window)
            subtracted = frame - background
            if subtract_step:
                reference = float(np.mean(shape[wc]))
                height = float(np.mean(subtracted[wc]) / reference) if reference else 0.0
                subtracted = subtracted - height * shape
            else:
                height = 0.0
            heights.append(height)
            l3_values.append(_integrate(energy, subtracted, w3))
            l2_values.append(_integrate(energy, subtracted, w2))
            sigma = _noise(energy, subtracted, fit_window)
            errors.append(sigma * step_size if np.isfinite(sigma) else np.nan)
            if np.isfinite(sigma) and sigma > 0:
                snr3.append(float(np.nanmax(subtracted[w3])) / sigma)
                snr2.append(float(np.nanmax(subtracted[w2])) / sigma)
            else:
                snr3.append(np.nan)
                snr2.append(np.nan)

        l3_values = np.array(l3_values)
        l2_values = np.array(l2_values)
        sigma = np.array(errors)
        e3 = sigma * np.sqrt(w3.sum())
        e2 = sigma * np.sqrt(w2.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = l3_values / l2_values
            error = np.abs(ratio) * np.sqrt(
                (e3 / l3_values) ** 2 + (e2 / l2_values) ** 2
            )
        snr2 = np.array(snr2)
        out[label] = {
            "l3": l3_values,
            "l2": l2_values,
            "ratio": ratio,
            "error": error,
            "step_height": np.array(heights),
            "l3_snr": np.array(snr3),
            "l2_snr": snr2,
        }
        median_snr = float(np.nanmedian(snr2)) if np.isfinite(snr2).any() else np.nan
        if np.isfinite(median_snr) and median_snr < min_snr:
            _logger.warning(
                "%s: the L2 white line is only %.1f sigma above noise per frame "
                "(< %.1f). The L3/L2 ratio is not measurable at this time "
                "resolution -- bin frames, or treat the ratio as meaningless.",
                label,
                median_snr,
                min_snr,
            )
    return out


def _white_line_model(energy, log_a, r, height, a3, c3, s3, a2, c2, s2,
                      step_width=1.0, step_ratio=2.0):
    """Power law + double arctan step + two Gaussian white lines."""
    background = np.exp(log_a) * energy**-r
    step = height * _double_step(energy, (c3, c2), step_width, step_ratio)
    peak3 = a3 / (s3 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((energy - c3) / s3) ** 2)
    peak2 = a2 / (s2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((energy - c2) / s2) ** 2)
    return background + step + peak3 + peak2


def fit_white_lines(
    spectra,
    *,
    fit_range=(800.0, 940.0),
    onsets=(855.0, 872.4),
    step_width=1.0,
    step_ratio=2.0,
    widths=(1.2, 1.5),
):
    """L3/L2 by fitting background, step and both peaks together.

    :func:`white_line_ratio` fits a power law on a pre-edge window and
    extrapolates it across the edge.  That extrapolation is the weak point:
    a small error in the exponent becomes a large error 100 eV away, and it
    lands almost entirely on L2, which is the small integral.  On this data it
    left the region *between* the two white lines sitting 3 sigma below zero,
    where a correct subtraction would put it at zero.

    Fitting the power law, the continuum step and both Gaussian white lines
    simultaneously over one range extrapolates nothing: every component is
    constrained by the same data, and the ratio comes from fitted peak areas
    rather than window sums.  The fit residual is returned so the model can be
    checked instead of trusted.

    Returns ``{label: {'ratio', 'error', 'l3_area', 'l2_area', 'centres',
    'widths', 'residual_sd', 'inter_peak_sd', 'success'}}``, each over frames.
    """
    from scipy.optimize import curve_fit

    out = {}
    for label, energy, data in _unpack(spectra):
        inside = _window(energy, fit_range)

        ratios, errors, l3s, l2s = [], [], [], []
        centres, sigmas, resid_sd, gap_sd, ok = [], [], [], [], []
        for frame in data:
            y_all = frame[inside]
            # Aligning with expand=True pads the ends of the axis with NaN, and
            # curve_fit refuses any non-finite value, so drop them rather than
            # failing on every frame of an aligned signal.
            finite = np.isfinite(y_all)
            if finite.sum() < 20:
                _logger.warning(
                    "%s: only %d finite channels in %s eV; skipping",
                    label, int(finite.sum()), fit_range,
                )
                for store in (ratios, errors, l3s, l2s, resid_sd, gap_sd):
                    store.append(np.nan)
                centres.append((np.nan, np.nan))
                sigmas.append((np.nan, np.nan))
                ok.append(False)
                continue
            x, y = energy[inside][finite], y_all[finite]
            gap = (x > onsets[0] + 5) & (x < onsets[1] - 4)
            # Starting guess: power law from the pre-edge, peaks from the data.
            pre = x < onsets[0] - 5
            try:
                slope, intercept = np.polyfit(
                    np.log(x[pre][y[pre] > 0]), np.log(y[pre][y[pre] > 0]), 1
                )
            except (ValueError, TypeError, np.linalg.LinAlgError):
                slope, intercept = -3.0, np.log(max(np.nanmean(y), 1e-9))
            background = np.exp(intercept) * x**slope
            amplitude = max(np.nanmax(y - background), 1e-6)
            guess = [intercept, -slope, abs(np.nanmedian((y - background)[x > onsets[1] + 20])),
                     amplitude * 3.0, onsets[0], widths[0],
                     amplitude * 0.8, onsets[1], widths[1]]

            def model(e_, log_a, r_, h_, a3_, c3_, s3_, a2_, c2_, s2_):
                return _white_line_model(e_, log_a, r_, h_, a3_, c3_, s3_,
                                         a2_, c2_, s2_, step_width, step_ratio)

            bounds = (
                [guess[0] - 10, 0.0, -np.inf, 0.0, onsets[0] - 3, 0.3,
                 0.0, onsets[1] - 3, 0.3],
                [guess[0] + 10, 10.0, np.inf, np.inf, onsets[0] + 3, 6.0,
                 np.inf, onsets[1] + 3, 6.0],
            )
            # The pre-edge guess can land outside its bounds -- a negative
            # power-law exponent, say, when the continuum step's tail reaches
            # into the fit window -- and curve_fit refuses to start there.
            guess = [
                float(np.clip(value, low + 1e-9, high - 1e-9))
                if np.isfinite(value) else float(np.clip(0.0, low, high))
                for value, low, high in zip(guess, *bounds)
            ]
            try:
                popt, pcov = curve_fit(model, x, y, p0=guess, bounds=bounds,
                                       maxfev=20000)
                residual = y - model(x, *popt)
                a3, a2 = popt[3], popt[6]
                e3 = np.sqrt(max(pcov[3, 3], 0.0))
                e2 = np.sqrt(max(pcov[6, 6], 0.0))
                ratio = a3 / a2 if a2 else np.nan
                error = (abs(ratio) * np.sqrt((e3 / a3) ** 2 + (e2 / a2) ** 2)
                         if a3 and a2 else np.nan)
                ratios.append(ratio)
                errors.append(error)
                l3s.append(a3)
                l2s.append(a2)
                centres.append((popt[4], popt[7]))
                sigmas.append((popt[5], popt[8]))
                sd = float(np.std(residual))
                resid_sd.append(sd)
                gap_sd.append(float(np.mean(residual[gap]) / sd) if sd else np.nan)
                ok.append(True)
            except (RuntimeError, ValueError) as error:
                _logger.warning("%s: white-line fit failed (%s)", label, error)
                for store in (ratios, errors, l3s, l2s, resid_sd, gap_sd):
                    store.append(np.nan)
                centres.append((np.nan, np.nan))
                sigmas.append((np.nan, np.nan))
                ok.append(False)

        out[label] = {
            "ratio": np.array(ratios),
            "error": np.array(errors),
            "l3_area": np.array(l3s),
            "l2_area": np.array(l2s),
            "centres": np.array(centres),
            "widths": np.array(sigmas),
            "residual_sd": np.array(resid_sd),
            "inter_peak_sd": np.array(gap_sd),
            "success": np.array(ok),
        }
        bad = float(np.nanmedian(np.abs(out[label]["inter_peak_sd"])))
        if np.isfinite(bad) and bad > 2.0:
            _logger.warning(
                "%s: the fit leaves the region between the white lines %.1f sigma "
                "from zero -- the model does not describe this spectrum, so the "
                "ratio is unreliable.",
                label, bad,
            )
    return out


def hyperspy_white_lines(
    spectra,
    *,
    fit_range=(820.0, 900.0),
    onsets=(855.0, 872.4),
    widths=(1.0, 1.2),
    step=True,
    step_width=1.0,
    step_ratio=2.0,
    show_progressbar=False,
):
    """L3/L2 from a HyperSpy model: power law + two Gaussians (+ optional step).

    A deliberately small model, fitted over a window around the edge only.

    **Why not eXSpy's ``EELSCLEdge`` components?**  They were tried and are the
    wrong tool for this measurement: HyperSpy ties an element's L3, L2 and L1
    intensities to a single parameter (right for quantification, since the
    subshell populations are fixed), so the GOS edges *cannot* represent a
    non-statistical white-line ratio.  Adding Gaussians on top then leaves the
    model degenerate -- on this data the components ended up summing to several
    times the signal and cancelling, with the power-law exponent pinned at its
    bound.  Use eXSpy's edges for quantification; use this for L3/L2.

    Parameters
    ----------
    fit_range : (float, float)
        Fit window.  Keep it close around the edge: a power law describes the
        background well over tens of eV, much less so over hundreds.
    step : bool
        Include a double arctan continuum step, tied 2:1 at L3:L2 from the
        4:2 degeneracy.  **Leave this on.**  The step sits under both white
        lines but is a far larger fraction of the small L2, so dropping it
        biases the ratio upwards; the difference is reported on this data in
        the package README.
    step_width, step_ratio :
        Arctan width in eV and the L3:L2 step height ratio.

    Returns ``{label: {'ratio', 'error', 'l3_area', 'l2_area', 'centres',
    'widths', 'step_height', 'residual_sd', 'model'}}``.
    """
    import hyperspy.api as hs

    out = {}
    for label, energy, data in _unpack(spectra):
        source = spectra[label] if hasattr(spectra, "keys") else spectra
        signal = hs.signals.Signal1D(
            np.asarray(source.data, dtype=float),
            axes=[a.get_axis_dictionary() for a in source.axes_manager._axes],
        )
        model = signal.create_model()
        model.set_signal_range(*fit_range)

        background = hs.model.components1D.PowerLaw()
        background.name = "background"
        model.append(background)

        step_component = None
        if step:
            step_component = hs.model.components1D.Expression(
                expression=(
                    "h*(ratio*(atan((x-c3)/w)/pi + 0.5) + "
                    "(atan((x-c2)/w)/pi + 0.5))/(ratio + 1)"
                ),
                name="step",
                h=float(np.nanmedian(data)) * 0.1,
                c3=float(onsets[0]),
                c2=float(onsets[1]),
                w=float(step_width),
                ratio=float(step_ratio),
            )
            step_component.w.free = False
            step_component.ratio.free = False
            step_component.c3.free = False
            step_component.c2.free = False
            model.append(step_component)

        peaks = []
        inside = _window(energy, fit_range)
        guess_area = float(
            np.nanmax(data[..., inside]) - np.nanmedian(data[..., inside])
        ) * max(widths[0], 0.5)
        for centre, width, name in zip(onsets, widths, ("L3", "L2")):
            peak = hs.model.components1D.Gaussian()
            peak.name = name
            peak.centre.value = float(centre)
            peak.centre.bmin, peak.centre.bmax = centre - 3.0, centre + 3.0
            peak.sigma.value = float(width)
            peak.sigma.bmin, peak.sigma.bmax = 0.3, 6.0
            peak.A.value = max(guess_area, 1e-9)
            peak.A.bmin = 0.0
            model.append(peak)
            peaks.append(peak)

        model.multifit(bounded=True, show_progressbar=show_progressbar)

        def values(parameter, field=None):
            try:
                signal_out = parameter.as_signal(field=field) if field else parameter.as_signal()
                return np.asarray(signal_out.data, dtype=float).ravel()
            except Exception:
                return np.full(data.shape[0], np.nan)

        a3, a2 = values(peaks[0].A), values(peaks[1].A)
        s3, s2 = values(peaks[0].A, "std"), values(peaks[1].A, "std")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = a3 / a2
            error = np.abs(ratio) * np.sqrt((s3 / a3) ** 2 + (s2 / a2) ** 2)

        model_data = model.as_signal(show_progressbar=show_progressbar).data
        diff = np.atleast_2d(signal.data)[..., inside] - np.atleast_2d(model_data)[..., inside]
        residual = np.nanstd(diff, axis=-1).ravel()

        out[label] = {
            "ratio": ratio,
            "error": error,
            "l3_area": a3,
            "l2_area": a2,
            "centres": np.stack([values(peaks[0].centre), values(peaks[1].centre)], axis=1),
            "widths": np.stack([values(peaks[0].sigma), values(peaks[1].sigma)], axis=1),
            "step_height": values(step_component.h) if step_component else np.zeros_like(a3),
            "residual_sd": residual,
            "model": model,
        }
        bad = int(np.sum(~np.isfinite(ratio)))
        if bad:
            _logger.warning("%s: %d frame(s) failed to fit", label, bad)
    return out


def bin_frames(spectra, n, *, times=None):
    """Average every ``n`` consecutive frames, to trade time for signal.

    Returns ``(binned spectra, binned times)``; ``times`` may be ``None``.
    Noise falls as sqrt(n), so binning 5 of these 18 s frames gives 90 s
    resolution with roughly half the scatter.
    """
    import hyperspy.api as hs

    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    binned = {}
    for label, energy, data in _unpack(spectra):
        usable = (data.shape[0] // n) * n
        if usable == 0:
            raise ValueError(f"{data.shape[0]} frames cannot be binned by {n}")
        if usable != data.shape[0]:
            _logger.warning(
                "%s: dropping the last %d frame(s), not a whole bin",
                label,
                data.shape[0] - usable,
            )
        block = data[:usable].reshape(-1, n, data.shape[1]).mean(axis=1)
        source = spectra[label] if hasattr(spectra, "keys") else spectra
        time_axis = source.axes_manager._axes[0]
        signal = hs.signals.Signal1D(
            block,
            axes=[
                {
                    "name": time_axis.name,
                    "size": block.shape[0],
                    "scale": getattr(time_axis, "scale", 1.0) * n,
                    "offset": getattr(time_axis, "offset", 0.0),
                    "units": time_axis.units,
                    "navigate": True,
                },
                {
                    "name": source.axes_manager.signal_axes[0].name,
                    "size": energy.size,
                    "scale": source.axes_manager.signal_axes[0].scale,
                    "offset": source.axes_manager.signal_axes[0].offset,
                    "units": source.axes_manager.signal_axes[0].units,
                    "navigate": False,
                },
            ],
        )
        signal.metadata.General.title = f"{label} (binned x{n})"
        binned[label] = signal

    if times is None:
        return binned, None
    times = np.asarray(times, dtype=float)
    usable = (times.size // n) * n
    return binned, times[:usable].reshape(-1, n).mean(axis=1)


def zlp_position(low_loss, half_width=1.5):
    """Sub-pixel ZLP peak position per navigation point, by parabolic refinement.

    Deliberately a *peak* position over a narrow window, not a centroid: a
    centroid taken over a few eV is pulled around by the plasmon tail, which
    scales with thickness, so it reports specimen structure as if it were
    energy jitter.  Measured that way the apparent jitter on this data was
    120 meV; the true peak jitter is 96 meV.

    Returns a ``(frames, y, x)`` array in eV.
    """
    window = low_loss.isig[-float(half_width) : float(half_width)]
    axis = window.axes_manager.signal_axes[0].axis
    data = np.asarray(
        window.data.compute() if getattr(window, "_lazy", False) else window.data
    )
    index = np.argmax(data, axis=-1).clip(1, data.shape[-1] - 2)

    def take(offset):
        return np.take_along_axis(data, (index + offset)[..., None], -1)[..., 0]

    low, middle, high = take(-1), take(0), take(1)
    denominator = low - 2 * middle + high
    delta = np.where(
        np.abs(denominator) > 0,
        0.5 * (low - high) / np.where(denominator == 0, 1, denominator),
        0.0,
    )
    return axis[index] + np.clip(delta, -1, 1) * float(axis[1] - axis[0])


def align_to_zlp(
    low_loss,
    *also_align,
    signal_range=(-4.0, 4.0),
    calibrate=True,
    crop=False,
    expand=True,
    subpixel=True,
    show_progressbar=False,
):
    """Align the low-loss on its zero-loss peak and shift the high-loss to match.

    Wraps eXSpy's ``align_zero_loss_peak(also_align=...)``, which is built for
    exactly this: the shifts are measured on the sharp, strong ZLP and applied
    unchanged to the other signals of the same acquisition.  Both stay lazy.

    On this data the ZLP sits **+214 meV off zero** and jitters pixel to pixel
    with a 96 meV standard deviation (1.1-1.4 eV peak to peak).  Aligning drops
    that to 39 meV and re-zeroes the axis, which is worth doing for the energy
    scale -- a 214 meV error applies to every edge onset you quote.

    It will **not** noticeably sharpen the core-loss edges: 96 meV of jitter
    added in quadrature to the Ni L3's 780 meV width changes it by under 1%.
    Align for calibration and for low-loss work (band gap, plasmons), not
    expecting a better white-line ratio.

    It is not cheap: the series is read twice (once to measure, once to shift),
    about 15-20 minutes for a 50-frame dual-EELS pair here.  ``expand=False``
    passes through once but leaves a larger residual misalignment.

    Parameters
    ----------
    low_loss : EELS signal
        The low-loss series; shifts are measured on this.
    *also_align : EELS signals
        Usually the high-loss. Shifted by the same amounts, in place.
    calibrate : bool
        Put the ZLP at 0 eV, correcting the absolute energy scale.
    crop, expand :
        Shifting leaves the ends undefined. ``crop=True`` trims them;
        ``expand=True`` pads instead, keeping every channel.

    Returns
    -------
    ``(low_loss, *also_align, report)`` -- the signals are modified in place
    and also returned, and ``report`` holds the before/after ZLP statistics.
    """
    before = zlp_position(low_loss)
    low_loss.align_zero_loss_peak(
        also_align=list(also_align),
        signal_range=tuple(signal_range),
        calibrate=calibrate,
        crop=crop,
        expand=expand,
        subpixel=subpixel,
        print_stats=False,
        show_progressbar=show_progressbar,
    )
    after = zlp_position(low_loss)

    report = {
        "offset_before_eV": float(np.nanmean(before)),
        "offset_after_eV": float(np.nanmean(after)),
        "jitter_before_eV": float(np.nanstd(before)),
        "jitter_after_eV": float(np.nanstd(after)),
        "dispersion_eV": float(low_loss.axes_manager.signal_axes[0].scale),
        "shifts_eV": before,
    }
    _logger.info(
        "ZLP: offset %+.0f -> %+.0f meV, pixel jitter %.0f -> %.0f meV "
        "(%.2f -> %.2f channels)",
        1000 * report["offset_before_eV"],
        1000 * report["offset_after_eV"],
        1000 * report["jitter_before_eV"],
        1000 * report["jitter_after_eV"],
        report["jitter_before_eV"] / report["dispersion_eV"],
        report["jitter_after_eV"] / report["dispersion_eV"],
    )
    return (low_loss,) + tuple(also_align) + (report,)


def fixed_pattern(reference, *, smooth_window=51, polyorder=2, scale=1.0):
    """Estimate the instrumental fixed pattern from a featureless region.

    An imperfect dark reference leaves a per-channel offset that is the same in
    every frame.  It looks exactly like shot noise -- high frequency in energy
    -- but it never averages down, so frame averaging stalls: on this data the
    residual fell as ``n**-0.35`` instead of ``n**-0.5``, and no amount of
    binning got the Ni L2 white line above ~5 sigma.

    Pass a region with no edges over the range you care about (the vacuum
    region of the same acquisition is ideal, and being from the same frames it
    carries the same pattern -- measured here at r = +0.86 against the
    specimen region's).  The pattern is the frame-averaged spectrum minus a
    smooth fit to it.

    It is **additive**, not multiplicative: dividing by a gain estimated the
    same way made things far worse here, because the reference is near zero in
    places and the division amplifies its noise.

    Parameters
    ----------
    reference : signal or dict with one entry
        The featureless region's per-frame spectra.
    smooth_window, polyorder :
        Savitzky-Golay parameters for the smooth component.  The window must be
        wide enough to pass over the pattern but narrow enough to follow the
        real spectrum.
    scale : float
        Multiply the pattern before returning it, if the reference and target
        regions see it at different amplitudes.

    Returns
    -------
    ``(channels,)`` array to pass to :func:`remove_fixed_pattern`.
    """
    from scipy.signal import savgol_filter

    items = _unpack(reference)
    if len(items) != 1:
        raise ValueError(
            f"give exactly one reference region, got {[k for k, _, _ in items]}"
        )
    _, energy, data = items[0]
    averaged = np.nanmean(data, axis=0)
    window = int(smooth_window)
    if window % 2 == 0:
        window += 1
    if window >= averaged.size:
        raise ValueError("smooth_window is wider than the spectrum")
    smooth = savgol_filter(averaged, window, int(polyorder))
    pattern = (averaged - smooth) * float(scale)

    # A sharp edge is narrower than the smoothing window, so it lands in the
    # "pattern" instead of the smooth part and corrupts the estimate.  Look for
    # an excursion several channels WIDE: a single hot channel is a cosmic ray
    # or a dead pixel, which is exactly what this is meant to remove, whereas a
    # real edge is broad enough to span many channels.
    spread = float(np.nanstd(pattern))
    if spread > 0:
        loud = np.abs(pattern) > 8.0 * spread
        if loud.any():
            runs, length, start = [], 0, 0
            for index, flag in enumerate(loud):
                if flag:
                    length = length + 1 if length else 1
                    start = index - length + 1
                elif length:
                    runs.append((length, start))
                    length = 0
            if length:
                runs.append((length, start))
            widest, where = max(runs)
            if widest >= 3:
                _logger.warning(
                    "the reference has a %d-channel excursion near %.0f eV: it is "
                    "not featureless there, so the pattern estimate is "
                    "contaminated. Use a region with no edges (vacuum), or widen "
                    "smooth_window.",
                    widest, energy[where + widest // 2],
                )
    _logger.info(
        "fixed pattern: sd %.3e against a reference level of %.3e (%.1f%%)",
        np.nanstd(pattern),
        np.nanmean(np.abs(smooth)),
        100 * np.nanstd(pattern) / max(np.nanmean(np.abs(smooth)), 1e-30),
    )
    return pattern


def remove_fixed_pattern(spectra, pattern):
    """Subtract a :func:`fixed_pattern` from every frame of every region."""
    import hyperspy.api as hs

    pattern = np.asarray(pattern, dtype=float)
    out = {}
    single = hasattr(spectra, "axes_manager")
    source_map = {spectra.metadata.General.title: spectra} if single else spectra
    for label, energy, data in _unpack(spectra):
        if pattern.size != data.shape[1]:
            raise ValueError(
                f"pattern has {pattern.size} channels but {label!r} has {data.shape[1]}"
            )
        source = source_map[label]
        corrected = source.deepcopy()
        corrected.data = data - pattern[None, :]
        corrected.metadata.General.title = f"{label} (pattern removed)"
        out[label] = corrected
    return out


def rolling_frames(spectra, window, *, step=1, times=None, mode="mean"):
    """Moving-window average over frames: keeps sampling, gains signal.

    A moving window of ``w`` frames gives ``n - w + 1`` points instead of the
    ``n / w`` that :func:`bin_frames` gives, so the series stays finely
    sampled while each point has sqrt(w) less noise.

    **The extra points are not extra information.** Neighbouring windows share
    ``w - 1`` of their frames, so the points are strongly correlated and a
    significance test that treats them as independent will report a far too
    small p-value. The returned ``effective_n`` is the number of genuinely
    independent windows (``n / w``); pass it to :func:`trend`, which then
    widens the error and uses the right degrees of freedom.

    Returns ``(spectra, times, effective_n)``.
    """
    import hyperspy.api as hs

    window = int(window)
    step = int(step)
    if window < 1 or step < 1:
        raise ValueError("window and step must be >= 1")
    if mode not in ("mean", "sum"):
        raise ValueError(f"mode must be 'mean' or 'sum', got {mode!r}")

    out = {}
    starts = None
    for label, energy, data in _unpack(spectra):
        n_frames = data.shape[0]
        if window > n_frames:
            raise ValueError(f"window {window} exceeds {n_frames} frames")
        starts = list(range(0, n_frames - window + 1, step))
        block = np.stack(
            [data[start : start + window].mean(axis=0) for start in starts]
        )
        if mode == "sum":
            block = block * window

        source = spectra[label] if hasattr(spectra, "keys") else spectra
        time_axis = source.axes_manager._axes[0]
        signal_axis = source.axes_manager.signal_axes[0]
        signal = hs.signals.Signal1D(
            block,
            axes=[
                {
                    "name": time_axis.name,
                    "size": block.shape[0],
                    "scale": getattr(time_axis, "scale", 1.0) * step,
                    "offset": getattr(time_axis, "offset", 0.0)
                    + getattr(time_axis, "scale", 1.0) * (window - 1) / 2.0,
                    "units": time_axis.units,
                    "navigate": True,
                },
                {
                    "name": signal_axis.name,
                    "size": energy.size,
                    "scale": signal_axis.scale,
                    "offset": signal_axis.offset,
                    "units": signal_axis.units,
                    "navigate": False,
                },
            ],
        )
        signal.metadata.General.title = f"{label} (rolling {window})"
        out[label] = signal

    effective_n = max(2, int(round((starts[-1] + window) / window))) if starts else 2
    rolled_times = None
    if times is not None:
        times = np.asarray(times, dtype=float)
        rolled_times = np.array(
            [times[start : start + window].mean() for start in starts]
        )
    _logger.info(
        "rolling window %d step %d: %d points from %d frames, "
        "effective independent points %d",
        window, step, len(starts), len(starts) + window - 1, effective_n,
    )
    return out, rolled_times, effective_n


def trend(values, times=None, errors=None, *, effective_n=None):
    """Least-squares straight line through a series, with significance.

    Answers 'does this actually change across the stack, or is it noise?'
    ``p`` is the two-sided p-value for a non-zero slope; ``change`` is the
    fitted change from first to last point as a fraction of the mean.

    ``effective_n`` is for series whose points are correlated -- the output of
    :func:`rolling_frames`, where neighbouring windows share most of their
    frames.  Treating those as independent inflates the significance badly, so
    pass the ``effective_n`` that function returns: the error is widened by
    sqrt(n / effective_n) and the p-value uses ``effective_n - 2`` degrees of
    freedom.
    """
    from scipy import stats

    values = np.asarray(values, dtype=float)
    x = np.arange(values.size, dtype=float) if times is None else np.asarray(times, float)
    good = np.isfinite(values) & np.isfinite(x)
    if good.sum() < 3:
        return {"slope": np.nan, "stderr": np.nan, "p": np.nan,
                "change": np.nan, "n": int(good.sum())}
    fit = stats.linregress(x[good], values[good])
    span = x[good].max() - x[good].min()
    mean = np.nanmean(values[good])
    scatter = np.nanstd(values[good])

    n_used = int(good.sum())
    stderr = float(fit.stderr)
    p_value = float(fit.pvalue)
    if effective_n is not None:
        effective_n = max(3, int(effective_n))
        stderr = stderr * np.sqrt(n_used / effective_n)
        t_stat = fit.slope / stderr if stderr else np.inf
        p_value = float(2 * stats.t.sf(abs(t_stat), df=effective_n - 2))

    return {
        "slope": float(fit.slope),
        "stderr": stderr,
        "p": p_value,
        "r": float(fit.rvalue),
        "change": float(fit.slope * span / mean) if mean else np.nan,
        "mean": float(mean),
        "scatter": float(scatter),
        "rsd": float(scatter / mean) if mean else np.nan,
        "n": n_used,
        "effective_n": None if effective_n is None else int(effective_n),
    }


def edge_table(
    spectra,
    *,
    times=None,
    oxygen=((528.0, 568.0), (480.0, 525.0)),
    nickel=((850.0, 890.0), (800.0, 845.0)),
    white_lines=None,
    regions=None,
):
    """Assemble O K, Ni L and L3/L2 per frame per region into a DataFrame.

    ``oxygen`` and ``nickel`` are each ``(integration window, pre-edge fit
    window)``.  ``white_lines`` is passed to :func:`white_line_ratio`.
    ``regions`` restricts which keys of ``spectra`` are used -- skip the
    vacuum, where there is nothing to measure.

    Columns: ``region, frame, time_s, O_K, O_K_err, Ni_L, Ni_L_err,
    O_over_Ni, L3, L2, L3_L2, L3_L2_err, L2_snr``.  Check ``L2_snr`` before
    reading anything into ``L3_L2``.
    """
    import pandas as pd

    if regions is not None:
        spectra = {k: v for k, v in spectra.items() if k in regions}

    oxygen_result = edge_intensity(spectra, oxygen[0], oxygen[1])
    nickel_result = edge_intensity(spectra, nickel[0], nickel[1])
    ratio_result = white_line_ratio(spectra, **(white_lines or {}))

    rows = []
    for label in spectra:
        o = oxygen_result[label]
        n = nickel_result[label]
        w = ratio_result[label]
        n_frames = o["intensity"].size
        for frame in range(n_frames):
            with np.errstate(invalid="ignore", divide="ignore"):
                over = o["intensity"][frame] / n["intensity"][frame]
            rows.append(
                {
                    "region": label,
                    "frame": frame,
                    "time_s": np.nan if times is None else float(times[frame]),
                    "O_K": o["intensity"][frame],
                    "O_K_err": o["error"][frame],
                    "Ni_L": n["intensity"][frame],
                    "Ni_L_err": n["error"][frame],
                    "O_over_Ni": over,
                    "L3": w["l3"][frame],
                    "L2": w["l2"][frame],
                    "L3_L2": w["ratio"][frame],
                    "L3_L2_err": w["error"][frame],
                    # Read this before believing L3_L2: below ~3 the L2 white
                    # line is not detected and the ratio is meaningless.
                    "L2_snr": w["l2_snr"][frame],
                }
            )
    return pd.DataFrame(rows)


def plot_edge_series(table, *, columns=("O_K", "O_over_Ni", "L3_L2"), figsize=None):
    """Plot columns of :func:`edge_table` against time, one panel each.

    Each panel reports the fitted trend and whether it is significant, so a
    flat noisy series is not read as a change.
    """
    import matplotlib.pyplot as plt

    labels = {
        "O_K": "O K integral (counts/pixel)",
        "Ni_L": "Ni L integral (counts/pixel)",
        "O_over_Ni": "O K / Ni L",
        "L3_L2": "Ni L$_3$/L$_2$",
    }
    regions = list(dict.fromkeys(table["region"]))
    figsize = figsize or (4.7 * len(columns), 3.6)
    figure, axes = plt.subplots(
        1, len(columns), figsize=figsize, constrained_layout=True, squeeze=False
    )
    for axis, column in zip(axes[0], columns):
        for region in regions:
            part = table[table["region"] == region]
            x = part["time_s"].to_numpy()
            if not np.isfinite(x).any():
                x = part["frame"].to_numpy()
            y = part[column].to_numpy()
            error = part.get(f"{column}_err")
            line = axis.errorbar(
                x, y,
                yerr=None if error is None else error.to_numpy(),
                fmt=".-", ms=4, lw=0.9, elinewidth=0.6, capsize=0, label=region,
            )
            info = trend(y, x)
            if np.isfinite(info["p"]):
                axis.plot(
                    x, info["mean"] + info["slope"] * (x - np.nanmean(x)),
                    ls="--", lw=1.0, color=line[0].get_color(), alpha=0.8,
                )
                verdict = "significant" if info["p"] < 0.05 else "not significant"
                axis.annotate(
                    f"{region}: {100 * info['change']:+.0f}% over the stack, "
                    f"p={info['p']:.2g} ({verdict})",
                    xy=(0.02, 0.97 - 0.07 * regions.index(region)),
                    xycoords="axes fraction", fontsize=6.5, va="top",
                    color=line[0].get_color(),
                )
        axis.set_xlabel("time (s)")
        axis.set_ylabel(labels.get(column, column))
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="lower right")
    return figure


