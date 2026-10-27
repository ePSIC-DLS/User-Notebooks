"""Check edge intensities and white-line ratios against spectra built to order.

Synthesising the spectrum means the right answer is known exactly: a power-law
background, a double-arctan continuum step and two Gaussian white lines of
chosen area. If the recipe is implemented correctly it must recover the ratio
those areas were given, and the O K integral must track a concentration that is
ramped on purpose.

    python -m gatan_insitu.test_spectroscopy
"""

from __future__ import annotations

import sys

import hyperspy.api as hs
import numpy as np

from .spectroscopy import (
    _double_step,
    bin_frames,
    edge_intensity,
    edge_table,
    power_law_background,
    trend,
    white_line_ratio,
)

_results = []


def check(name, condition, detail=""):
    _results.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return condition


def gaussian(energy, centre, width, area):
    return area / (width * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * ((energy - centre) / width) ** 2
    )


def build(n_frames=12, ratio=3.8, o_ramp=0.4, noise=0.0, seed=1):
    """A Ni L2,3 + O K spectrum series with a known L3/L2 and a ramped O K."""
    rng = np.random.default_rng(seed)
    energy = np.arange(460.0, 1086.0, 0.18)

    frames = []
    truth_o = []
    for index in range(n_frames):
        scale = 1.0 + o_ramp * index / (n_frames - 1)       # O K grows linearly
        truth_o.append(scale)
        spectrum = 5.0e7 * energy**-3.0                      # power-law background
        spectrum += 0.9 * _double_step(energy, (855.0, 872.4), 1.0, 2.0)
        spectrum += gaussian(energy, 855.0, 1.1, 8.0 * ratio)   # L3
        spectrum += gaussian(energy, 872.4, 1.4, 8.0)           # L2
        spectrum += 0.35 * scale * _double_step(energy, (532.0, 532.0), 2.0, 1.0)
        spectrum += gaussian(energy, 540.0, 3.0, 6.0 * scale)   # O K peak
        if noise:
            spectrum = spectrum + rng.normal(0, noise, energy.size)
        frames.append(spectrum)

    signal = hs.signals.Signal1D(
        np.array(frames),
        axes=[
            {"name": "Time", "size": n_frames, "scale": 18.0, "offset": 0.0,
             "units": "s", "navigate": True},
            {"name": "Energy loss", "size": energy.size, "scale": 0.18,
             "offset": float(energy[0]), "units": "eV", "navigate": False},
        ],
    )
    signal.metadata.General.title = "synthetic"
    return {"synthetic": signal}, energy, np.array(truth_o)


def build_dual(n_frames=3, size=6, shift_sd=0.12, seed=5):
    """A dual-EELS pair with a known per-pixel energy shift planted in both.

    The low loss gets a zero-loss peak and a plasmon; the high loss a power-law
    background and the Ni L3/L2 white lines. Both are displaced by the *same*
    per-pixel shift, as a real spectrometer drift displaces them, so aligning
    on the low-loss ZLP must straighten the high-loss edges too.
    """
    rng = np.random.default_rng(seed)
    shifts = rng.normal(0.0, shift_sd, (n_frames, size, size))

    low_axis = np.arange(-10.0, 40.0, 0.18)
    high_axis = np.arange(800.0, 950.0, 0.18)
    low = np.empty((n_frames, size, size, low_axis.size))
    high = np.empty((n_frames, size, size, high_axis.size))

    for index in np.ndindex(n_frames, size, size):
        offset = shifts[index]
        low[index] = (gaussian(low_axis, 0.0 + offset, 0.15, 100.0)
                      + gaussian(low_axis, 20.0 + offset, 3.0, 15.0))
        high[index] = (2.0e8 * (high_axis - offset) ** -3.0
                       + gaussian(high_axis, 855.0 + offset, 0.8, 3.8)
                       + gaussian(high_axis, 872.4 + offset, 0.9, 1.0))

    def wrap(data, axis, name):
        signal = hs.signals.Signal1D(data, axes=[
            {"name": "Time", "size": n_frames, "scale": 18.0, "offset": 0.0,
             "units": "s", "navigate": True},
            {"name": "y", "size": size, "scale": 1.0, "offset": 0.0,
             "units": "nm", "navigate": True},
            {"name": "x", "size": size, "scale": 1.0, "offset": 0.0,
             "units": "nm", "navigate": True},
            {"name": "Energy loss", "size": axis.size, "scale": 0.18,
             "offset": float(axis[0]), "units": "eV", "navigate": False},
        ])
        signal.set_signal_type("EELS")
        signal.set_microscope_parameters(beam_energy=200, convergence_angle=28.5,
                                         collection_angle=52.3)
        signal.metadata.General.title = name
        return signal

    return wrap(low, low_axis, "LL"), wrap(high, high_axis, "HL"), shifts


def check_alignment():
    from .spectroscopy import align_to_zlp, fit_white_lines, zlp_position

    low, high, planted = build_dual()
    print("\nZLP alignment, with shifts planted in both signals:")

    measured = zlp_position(low)
    check("zlp_position recovers the planted shifts",
          np.corrcoef(measured.ravel(), planted.ravel())[0, 1] > 0.95,
          f"r = {np.corrcoef(measured.ravel(), planted.ravel())[0, 1]:.3f}, "
          f"residual {1000 * np.std(measured - planted):.0f} meV")

    # A centroid over a wide window would be dragged by the plasmon; the peak
    # position must not be.  Same spectra, plasmon doubled.
    louder = low.deepcopy()
    axis = louder.axes_manager.signal_axes[0].axis
    louder.data = louder.data + gaussian(axis, 20.0, 3.0, 15.0)[None, None, None, :]
    drift = np.nanmean(zlp_position(louder) - measured)
    check("zlp_position is not dragged by the plasmon tail",
          abs(drift) < 0.01, f"moved {1000 * drift:+.1f} meV when the plasmon doubled")

    aligned_low, aligned_high, report = align_to_zlp(
        low.deepcopy(), high.deepcopy(), expand=True)

    check("alignment removes the low-loss jitter",
          report["jitter_after_eV"] < 0.4 * report["jitter_before_eV"],
          f"{1000 * report['jitter_before_eV']:.0f} -> "
          f"{1000 * report['jitter_after_eV']:.0f} meV")
    check("alignment re-zeroes the low-loss axis",
          abs(report["offset_after_eV"]) < 0.5 * report["dispersion_eV"],
          f"{1000 * report['offset_before_eV']:+.0f} -> "
          f"{1000 * report['offset_after_eV']:+.0f} meV")

    # The point of also_align: the HIGH loss must be straightened too.
    before_centres = fit_white_lines({"HL": high})["HL"]["centres"][:, 0]
    after_centres = fit_white_lines({"HL": aligned_high})["HL"]["centres"][:, 0]
    spread_before = float(np.nanstd(before_centres))
    spread_after = float(np.nanstd(after_centres))
    check("the planted shift is visible in the unaligned high loss",
          spread_before > 0.5 * np.std(planted),
          f"L3 centre scatter {1000 * spread_before:.0f} meV against "
          f"{1000 * np.std(planted):.0f} meV planted")
    check("also_align straightens the high-loss edges too",
          spread_after < 0.5 * spread_before,
          f"{1000 * spread_before:.0f} -> {1000 * spread_after:.0f} meV")
    check("the high-loss energy axis moved with the low-loss",
          abs(aligned_high.axes_manager.signal_axes[0].offset
              - high.axes_manager.signal_axes[0].offset) > 1e-6,
          f"{high.axes_manager.signal_axes[0].offset:.3f} -> "
          f"{aligned_high.axes_manager.signal_axes[0].offset:.3f} eV")
    check("both signals keep their shape", (
        aligned_low.data.shape[:3] == low.data.shape[:3]
        and aligned_high.data.shape[:3] == high.data.shape[:3]),
        f"{aligned_low.data.shape} / {aligned_high.data.shape}")


def main():
    true_ratio = 3.8
    spectra, energy, truth_o = build(ratio=true_ratio)
    times = np.arange(12) * 18.0
    print(f"synthetic series: 12 frames, true L3/L2 = {true_ratio}, O K ramped +40%")

    # --- power law ---
    background = power_law_background(energy, 5.0e7 * energy**-3.0, (800, 845))
    error = np.abs(background - 5.0e7 * energy**-3.0).max() / (5.0e7 * 1086.0**-3.0)
    check("power law recovers a pure power law", error < 1e-6,
          f"max error {error:.2e} of the signal at 1086 eV")
    check("power law returns NaN when the window has no positive data",
          not np.isfinite(power_law_background(energy, -np.ones_like(energy), (800, 845))).any())

    # --- white line ratio ---
    result = white_line_ratio(spectra)["synthetic"]
    recovered = float(np.median(result["ratio"]))
    check("L3/L2 recovered from a known spectrum",
          abs(recovered - true_ratio) / true_ratio < 0.05,
          f"{recovered:.3f} against a true {true_ratio}")
    check("the continuum step height is recovered",
          abs(np.median(result["step_height"]) - 0.9) / 0.9 < 0.05,
          f"{np.median(result['step_height']):.3f} against a true 0.9")

    # Skipping the step must bias the ratio DOWN -- the classic mistake.
    raw = white_line_ratio(spectra, subtract_step=False)["synthetic"]
    raw_ratio = float(np.median(raw["ratio"]))
    check("omitting the step biases the ratio downwards",
          raw_ratio < recovered * 0.9,
          f"{raw_ratio:.2f} without the step against {recovered:.2f} with it")

    # On a spectrum the step model fits exactly, the answer must not depend on
    # where the windows are put. (On real data it does, because the model is
    # imperfect and a wide window then picks up the residual between the
    # peaks -- which is why the windows are documented as "keep them tight".)
    spread = [
        float(np.median(white_line_ratio(spectra, l3=a, l2=b)["synthetic"]["ratio"]))
        for a, b in (((851, 859), (867.4, 877.4)), ((850, 860), (866, 878)),
                     ((852, 858), (869, 876)), ((844, 864), (864, 884)))
    ]
    check("ratio is window-independent when the step model is exact",
          max(abs(v - true_ratio) for v in spread) / true_ratio < 0.05,
          f"{[round(v, 2) for v in spread]} against a true {true_ratio}")

    # --- SNR reporting, the guard against measuring an undetected L2 ---
    clean = white_line_ratio(spectra)["synthetic"]
    noisy_spectra, _, _ = build(ratio=true_ratio, noise=0.02)
    noisy = white_line_ratio(noisy_spectra)["synthetic"]
    check("L2 SNR is reported and falls when noise is added",
          np.nanmedian(clean["l2_snr"]) > np.nanmedian(noisy["l2_snr"]) > 0,
          f"clean {np.nanmedian(clean['l2_snr']):.0f} sigma, "
          f"noisy {np.nanmedian(noisy['l2_snr']):.1f} sigma")
    check("L3 SNR exceeds L2 SNR, as the peaks require",
          np.nanmedian(clean["l3_snr"]) > np.nanmedian(clean["l2_snr"]))

    # --- O K intensity and its trend ---
    oxygen = edge_intensity(spectra, (528, 568), (480, 525))["synthetic"]
    measured = oxygen["intensity"] / oxygen["intensity"][0]
    expected = truth_o / truth_o[0]
    check("O K integral tracks the ramped concentration",
          np.abs(measured - expected).max() < 0.03,
          f"max error {np.abs(measured - expected).max():.3f} on a 40% ramp")

    info = trend(oxygen["intensity"], times)
    check("the O K ramp is detected as a significant change",
          info["p"] < 0.01 and info["change"] > 0.3,
          f"{100 * info['change']:+.0f}% over the stack, p={info['p']:.1e}")

    # L3/L2 was held constant while O K ramped, so it must not drift with it.
    # With literally zero noise any slope is "significant", so the meaningful
    # assertion is that the change is negligible in size.
    flat = trend(result["ratio"], times)
    check("a constant L3/L2 does not drift when O K ramps",
          abs(flat["change"]) < 0.02,
          f"{100 * flat['change']:+.2f}% over the stack")
    # Calibration of the significance test itself. Asserting p > 0.05 on one
    # random draw would fail 5% of the time by construction, so check the rate
    # over many draws instead: a correct test flags about 5% of pure noise.
    rng = np.random.default_rng(7)
    false_positives = sum(
        trend(rng.normal(0.0, 1.0, 30))["p"] < 0.05 for _ in range(400)
    )
    check("trend flags about 5% of pure-noise series, as it should",
          10 <= false_positives <= 35, f"{false_positives}/400 = "
          f"{100 * false_positives / 400:.1f}% (expect ~5%)")
    ramp = np.arange(30) * 0.25 + rng.normal(0.0, 1.0, 30)
    check("trend detects a real ramp buried in the same noise",
          trend(ramp)["p"] < 1e-6, f"p={trend(ramp)['p']:.1e}")

    # --- binning ---
    binned, binned_times = bin_frames(spectra, 3, times=times)
    check("binning reduces the frame count", binned["synthetic"].data.shape[0] == 4,
          str(binned["synthetic"].data.shape))
    check("binned time axis is rescaled",
          abs(binned["synthetic"].axes_manager["Time"].scale - 54.0) < 1e-9,
          f"{binned['synthetic'].axes_manager['Time'].scale} s")
    check("binned times are block means",
          np.allclose(binned_times, [18.0, 72.0, 126.0, 180.0]), str(binned_times))
    binned_ratio = float(np.median(white_line_ratio(binned)["synthetic"]["ratio"]))
    check("binning preserves the ratio",
          abs(binned_ratio - true_ratio) / true_ratio < 0.05,
          f"{binned_ratio:.3f}")

    noisy_binned, _ = bin_frames(noisy_spectra, 4, times=times)
    gain = (np.nanmedian(white_line_ratio(noisy_binned)["synthetic"]["l2_snr"])
            / np.nanmedian(noisy["l2_snr"]))
    check("binning x4 lifts the L2 SNR by about 2x", 1.5 < gain < 2.8,
          f"SNR x{gain:.2f}")

    # --- table ---
    table = edge_table(spectra, times=times)
    expected_columns = {"region", "frame", "time_s", "O_K", "O_K_err", "Ni_L",
                        "Ni_L_err", "O_over_Ni", "L3", "L2", "L3_L2",
                        "L3_L2_err", "L2_snr"}
    check("edge_table has the expected columns",
          expected_columns <= set(table.columns), str(sorted(set(table.columns))))
    check("edge_table rows match frames x regions", len(table) == 12, str(len(table)))
    check("edge_table L3/L2 matches white_line_ratio",
          np.allclose(table["L3_L2"].to_numpy(), result["ratio"], equal_nan=True))

    # --- model fitting, against the same known spectra ---
    print("\nmodel fitting (power law + step + two Gaussians):")
    from .spectroscopy import fit_white_lines, hyperspy_white_lines

    for engine, name in ((fit_white_lines, "scipy"),
                         (hyperspy_white_lines, "hyperspy")):
        recovered = []
        for true_value in (2.0, true_ratio, 6.0):
            known, _, _ = build(n_frames=3, ratio=true_value, noise=0.0)
            recovered.append(float(np.median(engine(known)["synthetic"]["ratio"])))
        worst = max(abs(got - want) / want
                    for got, want in zip(recovered, (2.0, true_ratio, 6.0)))
        check(f"{name} fit recovers known ratios", worst < 0.02,
              f"{[round(v, 3) for v in recovered]} against [2.0, {true_ratio}, 6.0]")

    scipy_value = float(np.median(fit_white_lines(spectra)["synthetic"]["ratio"]))
    hs_value = float(np.median(hyperspy_white_lines(spectra)["synthetic"]["ratio"]))
    check("the two independent implementations agree",
          abs(scipy_value - hs_value) / scipy_value < 0.02,
          f"scipy {scipy_value:.3f} vs hyperspy {hs_value:.3f}")

    no_step = float(np.median(
        hyperspy_white_lines(spectra, step=False)["synthetic"]["ratio"]))
    check("dropping the continuum step biases the ratio down",
          no_step < true_ratio * 0.85,
          f"{no_step:.3f} without the step against a true {true_ratio}")

    noisy_fit = fit_white_lines(noisy_spectra)["synthetic"]
    check("the fit reports an uncertainty that covers the truth",
          abs(np.nanmedian(noisy_fit["ratio"]) - true_ratio)
          < 3 * np.nanmedian(noisy_fit["error"]),
          f"{np.nanmedian(noisy_fit['ratio']):.3f} +/- "
          f"{np.nanmedian(noisy_fit['error']):.3f}")

    # --- fixed pattern and rolling window ---
    print("\nfixed pattern and rolling window:")
    from .spectroscopy import fixed_pattern, remove_fixed_pattern, rolling_frames

    rng2 = np.random.default_rng(3)
    # A channel-to-channel pattern, identical in every frame, as a dark-
    # reference residual is.  Alternating sign keeps it at a frequency the
    # smoothing cannot follow, which is what makes it separable at all.
    planted = 0.05 * ((-1.0) ** np.arange(energy.size)) * (
        1.0 + 0.3 * rng2.normal(0.0, 1.0, energy.size)
    )
    dirty = {k: v.deepcopy() for k, v in spectra.items()}
    dirty["synthetic"].data = spectra["synthetic"].data + planted[None, :] \
        + rng2.normal(0.0, 0.02, spectra["synthetic"].data.shape)

    # The reference must be featureless, as the vacuum region is -- a sharp
    # edge is narrower than the smoothing window and leaks into the pattern.
    flat = 5.0e7 * energy**-3.0
    reference_data = (flat[None, :] + planted[None, :]
                      + rng2.normal(0.0, 0.02, (12, energy.size)))
    reference = {"ref": hs.signals.Signal1D(
        reference_data,
        axes=[a.get_axis_dictionary()
              for a in spectra["synthetic"].axes_manager._axes],
    )}
    recovered_pattern = fixed_pattern(reference, smooth_window=51)
    correlation = np.corrcoef(recovered_pattern, planted)[0, 1]
    check("the planted fixed pattern is recovered from a featureless reference",
          correlation > 0.8, f"r = {correlation:.3f}")

    truth_mean = spectra["synthetic"].data.mean(axis=0)
    before = np.std(dirty["synthetic"].data.mean(axis=0) - truth_mean)
    cleaned = remove_fixed_pattern(dirty, recovered_pattern)
    after = np.std(cleaned["synthetic"].data.mean(axis=0) - truth_mean)
    check("removing it reduces the error against the truth", after < before * 0.7,
          f"{before:.4f} -> {after:.4f}")

    # Using a reference that does have edges must warn rather than mislead.
    import logging as _logging

    records = []
    handler = _logging.Handler()
    handler.emit = records.append
    logger = _logging.getLogger("gatan_insitu.spectroscopy")
    logger.addHandler(handler)
    fixed_pattern({"bad": dirty["synthetic"].deepcopy()}, smooth_window=51)
    logger.removeHandler(handler)
    check("a reference containing edges is flagged",
          any("featureless" in record.getMessage() for record in records),
          f"{len(records)} warning(s)")

    rolled, rolled_times, effective = rolling_frames(spectra, 4, times=times)
    check("rolling keeps the sampling", rolled["synthetic"].data.shape[0] == 9,
          f"{rolled['synthetic'].data.shape[0]} points from 12 frames, window 4")
    check("rolling reports the independent count", effective == 3, str(effective))
    check("rolling times are window centres",
          np.allclose(rolled_times[0], times[:4].mean()), f"{rolled_times[:2]}")

    # The correlation correction must tame an overlapping-window series.
    rng3 = np.random.default_rng(11)
    noise_only = rng3.normal(0.0, 1.0, 60)
    width = 10
    smoothed = np.array([noise_only[i:i + width].mean()
                         for i in range(len(noise_only) - width + 1)])
    naive_p = trend(smoothed)["p"]
    fixed_p = trend(smoothed, effective_n=len(noise_only) // width)["p"]
    check("overlapping windows inflate significance, and the correction undoes it",
          fixed_p > naive_p, f"naive p={naive_p:.4f} -> corrected p={fixed_p:.3f}")

    check_alignment()

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
