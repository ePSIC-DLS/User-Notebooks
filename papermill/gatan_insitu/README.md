# gatan_insitu

Lazy, fully calibrated loading of Gatan **multiframe (in-situ) STEM/EELS**
records into HyperSpy — the `.dm4` + `.raw` pairs that GMS writes for a
time-series spectrum image, including records split over several `.raw` files.

Developed and verified against `EELS_MF_data/InSitu (5)`: a dual-EELS
GIF Continuum K3 series, 50 frames × 118 × 111 probe positions × 3456 energy
channels, 9.05 GB per energy range.

## Quick start

```python
import gatan_insitu as gi

folder = r"EELS_MF_data/InSitu (5)"

# What's in here, without reading any bulk data:
print(gi.describe_folder(folder))

# Load everything lazily:
signals = gi.load_insitu_dataset(folder)
hl = signals["EELS HL"]      # <LazyEELSSpectrum, dimensions: (118, 111, 50|3456)>
ll = signals["EELS LL"]      # low-loss sibling, offset -36 eV
haadf = signals["HAADF Image"]
survey = signals["HAADF Image (SI Survey)"]

# Calibrated in all four axes, nothing read yet:
#   [nav] Time          50 @ 18.0381 s      (from the recorded frame timestamps)
#   [nav] y            111 @ 0.12708 nm
#   [nav] x            118 @ 0.12708 nm
#   [sig] Energy loss 3456 @ 0.18 eV, offset 464

ni = hl.isig[840.:900.].sum(axis="Energy loss")   # Ni L2,3 map per frame
ni.compute()                                      # ~1.5 s, reads 18% of the file
```

Run `python load_insitu_demo.py "<folder>"` for a worked example with timings,
or open **`gatan_insitu_example.ipynb`** for the notebook walkthrough.

### Notebook helpers

```python
gi.summary_table(folder)                      # the folder as a DataFrame
gi.quicklook(signals)                         # survey + first/last frame + spectra + trend
gi.make_navigator(hl, haadf=haadf)            # a real navigator, in 0.1 s
gi.computed(hl.isig[840.:900.].sum(axis="Energy loss"))   # compute AND return
```

## How the format works

A multiframe acquisition writes two files side by side:

| file | contents |
|---|---|
| `<name>.dm4` | the **last** frame of the series, plus the full tag tree |
| `<name>.raw` | every frame, headerless, back to back |

The tag subtree `ImageTags.In-situ.Raw File Format Info` describes the raw
stream exactly — dimensions, element type and size, frame size in bytes, and
each axis's calibration. Four things about it are worth knowing, all of them
verified against the frame embedded in the `.dm4` (see `verify.py`):

1. **The raw frame order is not the HyperSpy order.** DM lists dimensions as
   `Data0..DataN` with **`Data0` varying fastest**, so the C-order frame shape
   is that list *reversed*. For a spectrum image that means each frame is
   stored as `(E, y, x)` — 3456 consecutive *energy-slice images*, **not** a
   raster of spectra. So reading one pixel's spectrum is strided, while
   reading an energy window is contiguous. This drives the chunking (below).

2. **`Raw File Format Info` origins are already offsets in physical units**,
   unlike `ImageData.Calibrations`, which uses DM's index-origin convention
   (`offset = -Origin × Scale`). Both give 464.0 eV for the high-loss range
   here; this package uses the former directly.

3. **`Data Type` in the image header is not the DM *tag* type table.** Code
   `2` with `Data Size (bytes) = 4` means `float32`, not `int16`. The mapping
   in `DM_IMAGE_DTYPES` mirrors rosettasciio's own image-dtype table, and the
   loader cross-checks it against `Frame Size (bytes)`.

4. **Per-frame timestamps are 64-bit nanosecond counters split into 32-bit
   words**, low word first — so `Recorded.Sync.Per Frame.Time Since Start (ns)`
   is *twice* as long as the frame count. Decoded, this series runs at
   18.0387 ± 0.0049 s per frame, which is where the `Time` axis comes from
   rather than any nominal value.

## Multiple RAW files — they are one byte stream

Long recordings are split. Three naming conventions are recognised, with the
unnumbered file always first:

```
<stem>.raw   <stem>.raw_1   <stem>.raw_2      # GMS in-situ plugin
<stem>.raw   <stem>.raw.1   <stem>.raw.2
<stem>.raw   <stem>_1.raw   <stem>_2.raw
```

**GMS cuts at a fixed byte size — 10 GiB exactly — with no regard for frame
boundaries, so a frame routinely straddles two files.** From a real record:

```
STEM SI_EELS HL SI.raw     10737418240 B = 66 frames + 62525440 B
STEM SI_EELS HL SI.raw_1    7215810560 B = 44 frames + 99215360 B
                                           ^^^^^^^^   ^^^^^^^^
                                   these add up to exactly one 161740800 B frame
```

So the files are concatenated **logically** and the frame grid laid over the
result: `n_frames = total_bytes // frame_bytes` (111 here, matching the tagged
`# Frames`). Counting frames per file instead would drop the straddling frame
*and* misalign every frame after it — silent corruption that shows up as a
discontinuity partway along the energy axis, not as an error.

Chunks are always a single contiguous run in that stream, and `LazyRawArray`
resolves each one across however many files it spans. Splits need not be even
or aligned to anything; a boundary one byte into a frame works. A candidate
file that has its own `.dm4` beside it is skipped, so a sibling acquisition's
raw file is never adopted.

If a recording was aborted mid-frame, the partial final frame is dropped from
the *total* with a warning (`strict=True` raises instead), and a disagreement
with the tagged `# Frames` is reported with the file contents winning. Pass
`raw_paths=[...]` to override discovery entirely.

```bash
python -m gatan_insitu.test_multifile "<a dm4>"     # 8 checks
```

splits a real acquisition mid-frame — including boundaries one byte into and
one byte before a frame, and four files with three straddles — and compares
every frame against the unsplit load.

## Chunking, and why it is what it is

Chunks are cut along the axis that is **slowest in the file** and span whole
`y`/`x` planes, so every chunk is one contiguous run of bytes on disk. The
transpose into HyperSpy's `(…, E)` order happens afterwards in the dask graph
and costs nothing at read time.

The 16 MB default (`chunk_mb`) was measured on this dataset:

| `chunk_mb` | chunk | partitions | full 9 GB reduction | Ni window (840–900 eV) |
|---|---|---|---|---|
| 4 | 3.9 MB | 2200 | 4.2 s | 1.1 s, reads 11% |
| **16** | **15.7 MB** | **550** | **2.2 s** | **0.5 s, reads 18%** |
| 32 | 28.8 MB | 300 | 3.9 s | 1.0 s, reads 33% |
| 64 | 57.6 MB | 150 | 9.0 s | 1.5 s, reads 67% |

Bigger chunks cut the energy axis into fewer pieces, so a narrow energy window
drags in far more of the file; they were also slower outright here. Lower
`chunk_mb` further if you only ever want narrow windows.

### Per-pixel access is fine — do not rechunk for speed

The energy-sliced layout *sounds* bad for per-pixel work: reading one pixel's
spectrum touches all 11 energy chunks (~173 MB for 13.5 kB). In practice it is
not a problem, because **each chunk holds complete `y` and `x` planes**. So
when dask needs whole spectra it concatenates energy chunks *within one frame*
— a cheap local operation, not a global transpose.

Measured on an 8-frame, 700–1000 eV slice, against a spectrum-contiguous
`.hspy` copy of exactly the same data:

| operation | raw (energy-sliced) | `.hspy` gzip | `.hspy` uncompressed |
|---|---|---|---|
| one spectrum | **34 ms** | 117 ms | 73 ms |
| 20×20 ROI, all frames | **132 ms** | 830 ms | 342 ms |
| whole-spectrum rechunk + sum | **413 ms** | 2645 ms | 1394 ms |
| `remove_background` | **4.6 s** | 11.5 s | 7.2 s |

The raw stream wins every case, by 2–3× even against an uncompressed copy. So
**work directly on the lazy signal**; HyperSpy's per-pixel routines are fine on
it. (An earlier version of this README advised rechunking first. That advice
was reasoned from the layout rather than measured, and it was wrong.)

### When `save_rechunked` is still worth it

Not for speed — for **size and portability**. Cropping in energy and frames
turns a 9.05 GB `.raw` + `.dm4` pair into a single self-contained file, gzipped
by default:

```python
subset = hl.isig[700.:1000.]
gi.save_rechunked(subset, "hl_ni.hspy")   # 0.70 GB of data -> 0.044 GB on disk
```

Useful for sharing, archiving, or working away from the original storage, and
worth it whenever re-reading the raw is expensive (a network share). Pass
`compression=None` for a faster, much larger file.

It always passes explicit integer chunks, because HDF5 rejects the dask `-1`
shorthand (`OverflowError`), and it refuses `.zspy` with an explanation rather
than failing obscurely (rosettasciio 0.9 calls
`zarr.storage.NestedDirectoryStore`, removed in zarr 3). Call `s.close_file()`
before deleting a lazily loaded `.hspy` on Windows.

## Per-frame segmentation and masked spectra

Regions move and change shape through an in-situ series, so one mask drawn on
one frame is wrong for all the others. `segment_frames` thresholds **every
frame independently**; `masked_spectra` then sums the EELS signal inside each
region frame by frame, in a single pass over the data.

Segment the **full-resolution HAADF** and reduce the masks afterwards. The
HAADF is recorded at an integer multiple of the SI sampling (`SI.Scale`, 4×
here: 472×444 against 118×111), so this gives sharper boundaries than
segmenting a rebinned image, and a pixel that straddles a boundary can carry a
fractional weight instead of being forced into one class:

```python
signals = gi.load_insitu_dataset(folder)
hl, haadf = signals["EELS HL"], signals["HAADF Image"]

fine = gi.segment_frames(                       # on 472 x 444, per frame
    haadf, n_classes=3, smooth=4.0, min_size=64,
    labels=["vacuum", "Ni-rich", "Au-rich"],
)
masks = gi.rebin_masks(fine, target_shape=hl.data.shape[1:3])   # -> 118 x 111
spectra = gi.masked_spectra(hl, masks, mode="mean")             # one pass, ~8 s

spectra["Au-rich"]          # <EELSSpectrum, (50|3456)>, Time axis preserved
masks.areas                 # (50, 3) effective pixel counts
masks.plot(haadf, frames=(0, 16, 33, 49))      # see note below
masks.plot_areas(times=hl.metadata.InSitu.frame_times_s)
```

`masks.plot` takes a source at **any** resolution and maps the contours onto
its pixel grid, so SI-grid masks can be drawn over the full-resolution HAADF
they were cut from without applying the binning factor by hand. (Earlier
versions drew the contours in mask-pixel coordinates, which collapsed them into
the top-left corner, scaled down by exactly the binning factor.)

On this data 6.4% of SI pixels straddle a boundary and get split weights; the
reduced masks agree with segmenting the rebinned navigator to 99.5%, so either
works — the full-resolution route is just better defined. `mode="majority"`
gives boolean masks back if you want them.

Thresholds are recomputed per frame (multi-Otsu by default), which absorbs
drift in beam current or detector gain. `method="manual"` takes your own,
`threshold_smoothing=N` median-filters them over time to steady a jittery
series.

### Check the labels before you trust them

HAADF intensity scales as thickness × Z^n, so brightness alone **cannot tell
composition from thickness** — the thin rim of a heavy particle gives the same
grey level as a thick light one. A mask built on brightness is a hypothesis.

`compare_regions` tests it: it scales one region's mean spectrum onto another's
over an edge-free pre-edge window, where signal is pure background and so
proportional to thickness, then shows what is left.

```python
report = gi.compare_regions(
    spectra, "Ni-rich", "Au-rich", pre_window=(470, 520),
    edges={"O K": ((500, 525), (537, 562)), "Ni L": ((820, 845), (860, 885))},
)
report["flatness"]      # peak residual / peak signal
report["jump_ratios"]   # per region, thickness-free
```

A flat residual would mean the regions are the same material at different
thicknesses. On this dataset it is **25% of peak signal**, with the Ni L jump
ratio 1.07 in the grey region against 0.95 in the bright one — so the regions
really are compositionally distinct, and the Ni/Au labelling is supported.

`jump_ratio_map` gives a thickness-insensitive map to segment on instead, where
counts allow. On this series they do not: per pixel per frame the Ni ratio
separates the regions by ~0.13 against ~0.25 scatter even at `smooth=3`, so
masks cut from it wander into vacuum. Averaged over a region it is clean, which
is why it belongs in the *check* rather than the segmentation. Measure the
contrast-to-noise before relying on it — its docstring shows how.

```bash
python -m gatan_insitu.test_segmentation      # 26 checks, synthetic ground truth
```

builds a series whose regions drift and grow, and confirms the masks recover
the truth to 99.95% (against 80.7% for a single fixed mask), that extraction is
exact, and that `compare_regions` calls a pure scaling flat.

## Following an edge through the stack

`edge_table` measures O K, Ni L and the Ni L3/L2 white-line ratio for every
frame in every region, and `trend` says whether a series actually changes or is
just noise:

```python
table = gi.edge_table(spectra, times=hl.metadata.InSitu.frame_times_s,
                      regions=["Ni-rich", "Au-rich"])
gi.plot_edge_series(table)                       # O_K, O_over_Ni, L3_L2 vs time

part = table[table["region"] == "Ni-rich"]
gi.trend(part["O_K"].to_numpy(), part["time_s"].to_numpy())
# {'change': +0.229, 'p': 0.010, 'rsd': 0.186, ...}
```

`edge_intensity` fits a power law over a pre-edge window, subtracts it and
integrates. `white_line_ratio` is the Pearson method: power law, then a
**double arctan continuum step** with the 2:1 height ratio set by the 4:2
degeneracy of the L3 and L2 initial states. Omitting that step is the usual way
to get a wrong L3/L2 — on a synthetic spectrum with a true ratio of 3.8 the
recipe returns **3.78**, and skipping the step returns **2.18**.

`bin_frames(spectra, n)` trades time resolution for signal when an edge is too
weak per frame.

### Results on this dataset

**O K changes.** In the Ni-rich region the O K integral rises **+23% over the
900 s stack, p = 0.010**, and the result holds at every binning: +23.7% (×2),
+22.4% (×5, p = 0.002), +19.7% (×10). The Au-rich region shows no significant
change. This is a real trend.

**Ni L3/L2 is measurable, but only as a stack average** — and only after two
fixes. Window integration is not enough; see "Making L3/L2 work" below.

```python
pattern = gi.fixed_pattern({"vacuum": spectra["vacuum"]})      # instrumental
clean = gi.remove_fixed_pattern({"Ni-rich": spectra["Ni-rich"]}, pattern)
rolled, rolled_times, effective = gi.rolling_frames(clean, 15, times=times)
result = gi.hyperspy_white_lines(rolled)["Ni-rich"]
gi.trend(result["ratio"], rolled_times, effective_n=effective)   # p = 0.50, flat
```

Stack average: **L3/L2 = 4.0 ± 0.35**, i.e. Ni²⁺ (NiO is 3.5–4.5, Ni metal
~2). No significant change across the series at any window size.

### Aligning on the zero-loss peak

Dual-EELS records the low loss simultaneously, so the sharp, strong ZLP can be
used to measure the energy jitter and the **same shifts applied to the high
loss**. `align_to_zlp` wraps eXSpy's `align_zero_loss_peak(also_align=...)`,
which is built for exactly this; both signals stay lazy.

```python
ll, hl, report = gi.align_to_zlp(ll, hl)        # shifts measured on ll, applied to both
report["offset_before_eV"], report["jitter_before_eV"]
```

Measured on this series (all 50 frames):

| | value |
|---|---|
| ZLP offset from zero | **+255 meV** |
| per-pixel jitter (sd) | **97 meV** = 0.54 channels |
| peak to peak | 1061 meV |
| within-frame pixel scatter | 93 meV |
| frame-to-frame drift of the mean | 123 meV peak to peak |

The jitter is mostly **within** a frame, not drift between frames, so it has to
be corrected per pixel — a per-frame correction would miss most of it.
Aligning gives **offset +255 → −9 meV and jitter 97 → 31 meV** (0.54 → 0.17
channels), in 1547 s for the pair.

**Measure the peak, not a centroid.** `zlp_position` refines the peak
parabolically over ±1.5 eV. A centroid over a few eV is dragged by the plasmon
tail, which scales with thickness, so it reports specimen structure as energy
jitter — that route gave 120 meV where the true peak jitter is 97 meV.

**What it is actually worth**, measured on the Ni L edge of this series:

| | unaligned | aligned |
|---|---|---|
| L3 centre | 854.91 eV | **854.64 eV** (−269 meV) |
| L3 sigma | 0.777 eV | 0.803 eV (**+3.3%**) |
| L3/L2 | 3.99 ± 0.35 | 3.95 ± 0.30 |
| fit residual | 4.99e-4 | 4.43e-4 |

So: **align for the energy scale.** The axis was off by a quarter of an eV,
and that error sat on every onset — the Ni L3 really is at 854.6 eV, not
854.9 eV.

**Do not expect sharper edges.** The L3 got 3.3% *wider*, not narrower: 97 meV
of jitter in quadrature with a 780 meV width is under 1%, while the sub-pixel
interpolation smooths by slightly more than that. The improved residual is
partly the same smoothing — it also halved the fixed-pattern amplitude
(4.4e-4 → 2.6e-4), which flatters the fit without improving the physics. The
white-line ratio is unchanged within errors, as expected.

It costs a full extra pass: the series is read twice, ~25 minutes for a
50-frame dual-EELS pair here (`expand=False` reads once but leaves more
residual misalignment).

The test plants a known per-pixel shift in *both* signals of a synthetic
dual-EELS pair and checks the high loss is straightened by aligning on the low
loss: L3 centre scatter **107 meV → 9 meV**.

### Aligning the frames for specimen drift

`align_frames` measures the drift on the HAADF series (phase correlation, each frame against
the one before by default, so a changing specimen still matches) and applies it to every
stream of the acquisition:

```python
haadf_al, hl_al, ll_al, report = gi.align_frames(haadf, hl, ll)
report["total_drift_nm"], report["field_kept"], report["residual_rms_nm"]
```

The drift is rounded to **whole SI pixels** and every stream is **cropped** to the field common
to all frames: spectra are never interpolated, and because the HAADF is cropped by exactly 4x
the SI shift it still rebins exactly onto the aligned SI. The price is up to half an SI pixel
of residual misregistration, which the report gives. Everything stays lazy.
`python -m gatan_insitu.test_alignment` plants a sub-pixel, sign-changing drift that is not a
multiple of the 4x grid on an evolving specimen, and checks that every SI pixel then sees
the same specimen position in every frame (12 checks).

### Making L3/L2 work

Three things had to be fixed, each found by measurement:

**1. Window integration is the wrong estimator here.** These white lines are
only a few eV wide; a window wide enough to reach the dip between them drives
the small L2 integral towards zero and the ratio explodes (the first attempt
returned −29 ± 157, then 9.8 with tightened windows). The background is fitted
pre-edge and extrapolated ~100 eV, and that error lands almost entirely on L2.
Fitting instead — power law, continuum step and both Gaussians together over
820–900 eV — extrapolates nothing and gives **4.0**.

**2. A fixed-pattern detector residual was stalling frame averaging.** The
residual fell as `n**-0.35` rather than `n**-0.5`, and the frame-mean residual
variance was 3.5× what independence predicts. It is instrumental, not
specimen: the pattern in the Ni-rich region correlates **r = +0.86** with the
one in vacuum. Subtracting it restores ideal averaging:

| | scaling exponent | 50-frame noise |
|---|---|---|
| raw | −0.348 | 0.000523 |
| gain-corrected (divide) | −0.057 | 0.003308 |
| **offset-corrected (subtract)** | **−0.508** | **0.000272** |

It is additive, not multiplicative — dividing by a gain estimated the same way
made it far worse, because the reference is near zero in places.

**3. A moving window buys precision, not time resolution.** `rolling_frames`
keeps `n - w + 1` points instead of `n / w`, and the L3/L2 error falls from
±1.17 (single frames) to ±0.43 (window 25). But neighbouring windows share
`w - 1` frames, so **the points are correlated and a naive significance test
is badly wrong** — at window 15 the naive p is 0.001 while the corrected p is
0.50. Pass the `effective_n` that `rolling_frames` returns to `trend`, which
widens the error by `sqrt(n / effective_n)` and uses the right degrees of
freedom.

### Which estimator to use

| method | synthetic (true 3.8) | this data |
|---|---|---|
| `hyperspy_white_lines` (power law + step + 2 Gaussians) | **3.807** | **3.99 ± 0.35** |
| `fit_white_lines` (same model, scipy) | **3.801** | **4.18 ± 0.32** |
| `hyperspy_white_lines(step=False)` | 2.647 | 2.49 ± 0.24 |
| `white_line_ratio` (window integration) | 3.780 | 9.78 |

The two independent implementations agree to 0.2% on synthetic data and to
within their errors on the real data. **Keep the continuum step on** — it sits
under both white lines but is a far larger fraction of the small L2, so
dropping it biases the ratio down by ~30% on data where the truth is known.

eXSpy's `EELSCLEdge` components were tried and are the wrong tool for *this*
measurement: HyperSpy ties an element's L3, L2 and L1 intensities to one
parameter (correct for quantification, since the subshell populations are
fixed), so the GOS edges cannot represent a non-statistical white-line ratio.
Adding Gaussians on top left the model degenerate — components summing to
several times the signal and cancelling, with the power-law exponent pinned at
its bound. Use eXSpy's edges for quantification, these for L3/L2.

### What per-frame L3/L2 would need

Even after all of the above, a *trend* in L3/L2 stays out of reach. The tooling
says so rather than returning a number quietly:

```
Ni-rich: the L2 white line is only 3.0 sigma above noise per frame (< 3.0).
The L3/L2 ratio is not measurable at this time resolution -- bin frames, or
treat the ratio as meaningless.
```

The L2 white line is weak: 3.0σ per frame, and still only **4.9σ after
averaging all 50 frames**. Binning does not rescue it —

| binning | per point | L2 SNR | L3/L2 |
|---|---|---|---|
| ×1 | 18 s | 3.0 | 22.9 ± 80.2 |
| ×5 | 90 s | 3.9 | 14.4 ± 16.4 |
| ×10 | 180 s | 4.0 | 9.6 ± 2.4 |
| ×50 | 900 s | 4.9 | 9.2 |

With the pattern removed and a rolling window the ratio is stable (3.90 at
window 1 through 4.17 at window 25) and its error shrinks, but the corrected
p-value stays between 0.25 and 0.50 throughout: **no trend is detectable.** A
window wide enough for good L2 statistics leaves only 2–3 independent time
points out of the 900 s series. Resolving a *change* in Ni oxidation state here
would need more dose per frame, not more processing.

Always check `L2_snr` in the table before reading anything into `L3_L2`.

```bash
python -m gatan_insitu.test_spectroscopy      # 33 checks, synthetic ground truth
```

builds spectra with a known L3/L2 and a deliberately ramped O K, and confirms
both fit engines recover the ratio to 2% and agree with each other to 0.2%,
that omitting the step biases it down, that a planted fixed pattern is
recovered (r = 0.99) and its removal cuts the error 6×, that a reference
containing edges is flagged rather than silently used, that the O K ramp is
detected, and that `trend` flags ~5% of pure-noise series (5.8% over 400 draws)
while catching a real ramp — and that the overlapping-window correction undoes
the inflated significance it would otherwise report.

## The two layouts

```python
# layout="spectrum" (default): (time, y, x | E), LazyEELSSpectrum
hl = gi.load_insitu_signal(dm4)

# layout="native": (time, E | x, y), LazySignal2D — the on-disk order.
# Every navigation step is an energy-filtered image; no transpose at all.
ef = gi.load_insitu_signal(dm4, layout="native")
ef.plot()        # scrub through energy and time over the real images
```

## Other options

```python
gi.load_insitu_signal(
    dm4,
    chunk_mb=16,                     # see the table above
    frames=5,                        # int, slice or (start, stop) — quick looks
    time_origin="experiment_start",  # or "first_frame" (t - t[0])
    uniform_time=True,               # False -> non-uniform axis from the timestamps
    backend="memmap",                # or "fromfile" for network shares
    strict=False,                    # raise on a partial final frame
    lazy=True,                       # False computes — only with a frames subset
)
```

A `.dm4` with no in-situ tags (the SI survey image) falls through to
`hs.load()`, so `load_insitu_dataset` returns the whole folder uniformly.

## Metadata

Standard HyperSpy nodes are populated from the tags — beam energy (200 kV),
convergence (28.5 mrad) and collection (52.3 mrad) semi-angles, dwell time,
camera length, spectrometer name — and `set_microscope_parameters` is called
for EELS signals, so routines that read those parameters (`estimate_thickness`,
Kramers–Kronig analysis, quantification) find them already populated.

Multiframe-specific values live under a namespaced `InSitu` node:

```python
hl.metadata.InSitu.frame_times_s       # exact per-frame timestamps, 50 values
hl.metadata.InSitu.frame_interval_s    # 18.038
hl.metadata.InSitu.raw_files           # which .raw file(s) back this signal
hl.metadata.InSitu.raw_frame_layout    # 'Energy loss+y+x'
hl.metadata.InSitu.n_frames_declared   # the tagged count, for comparison
```

The complete DM tag tree is in `hl.original_metadata`, including the per-frame
spatial-drift corrections under
`ImageList.TagGroup1.ImageTags.SI.Acquisition.Artefact Correction.Spatial Drift`.

## Gotchas

- **`axes_manager[0]` is `x`, not `Time`.** HyperSpy lists navigation axes in
  reverse array order, so `hl.axes_manager` reads `(118, 111, 50|3456)` =
  `(x, y, time | E)`. Index by name — `axes_manager["Time"]` — and remember
  `hl.inav[i, j, k]` is `[x, y, time]`. Array order (`hl.data.shape`) is
  `(time, y, x, E)`.
- **Lazy `.compute()` works in place and returns `None`.** Write
  `m = s.sum(...); m.compute()`, not `m = s.sum(...).compute()`, or use
  `gi.computed(s)`, which computes and returns the signal.
- **Give `plot()` a navigator.** For a lazy signal HyperSpy builds one by
  summing a *single* chunk, so it shows ~315 of 3456 channels. The
  simultaneous HAADF is recorded at exactly 4× the SI sampling and rebins onto
  the navigation grid, so `gi.make_navigator(hl, haadf=haadf)` gives a real
  one in 0.1 s. For scrubbing, `layout="native"` moves in ~20 ms per step
  against ~150 ms in spectrum layout.
- The HAADF image is recorded at 4× the SI sampling (`SI.Scale = 4`):
  472 × 444 at 0.0318 nm against 118 × 111 at 0.1271 nm. The two are not
  pixel-aligned, so resample before overlaying.
- `da.from_array` calls `.copy()` on any array-like that has one, which for a
  bare `np.memmap` materialises the whole file in RAM. `raw_store.LazyRawArray`
  exists to avoid exactly that, and `verify.py` checks the dask graph for
  embedded bulk data so the trap cannot come back unnoticed.
- A loaded signal keeps its `.raw` files memory-mapped, so Windows will not let
  you move or delete them. Call `stream.close()` (reads reopen as needed):
  ```python
  from gatan_insitu.raw_store import LazyRawArray
  for v in hl.data.dask.values():
      if isinstance(v, LazyRawArray):
          v.stream.close()
  ```

## Verifying

Because a multiframe `.dm4` holds the last frame of its own series, the loader
can be checked against ground truth with no external reference:

```bash
python -m gatan_insitu.verify "EELS_MF_data/InSitu (5)"
```

This compares the final raw frame element-by-element with the frame
rosettasciio reads from the `.dm4`, cross-checks every axis calibration against
rosettasciio's, confirms both read backends agree, confirms the graph holds no
bulk data, and checks the frame bookkeeping and timestamp regularity.
All 35 checks pass on `InSitu (5)`.

Requires HyperSpy ≥ 2.0 and rosettasciio; tested with hyperspy 2.3.0,
rosettasciio 0.9.0, dask 2025.5.1, numpy 1.26.4.
