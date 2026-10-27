"""Per-frame segmentation of an in-situ series, and masked spectrum extraction.

Regions move and change shape through an in-situ series, so a mask drawn on
one frame is wrong for the rest.  :func:`segment_frames` thresholds **each
frame independently**, and :func:`masked_spectra` sums the EELS signal inside
each region, frame by frame, in a single pass over the data.

A caution about segmenting on HAADF intensity
---------------------------------------------
HAADF signal scales roughly as thickness x Z^n, so **intensity alone cannot
tell composition from thickness**: the thin rim of a heavy particle gives the
same grey level as a thick light one.  A mask built on HAADF brightness is a
hypothesis about composition, not a measurement of it.

So test it, with :func:`compare_regions`: scale one region's mean spectrum onto
another's using a pre-edge window, and look at the residual.  If the regions
differ only in thickness the spectra superimpose and the residual is flat; real
compositional differences leave edges standing in it.  :func:`jump_ratio_map`
then gives a thickness-insensitive alternative to segment on -- a post-edge
over pre-edge ratio, which is a composition proxy that needs no background fit
and is far steadier than power-law fitting on dark-corrected data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import dask.array as da
import numpy as np

_logger = logging.getLogger(__name__)

__all__ = [
    "MaskStack",
    "compare_regions",
    "composition_ratio",
    "erode_masks",
    "elemental_map",
    "jump_ratio_map",
    "masked_spectra",
    "rebin_masks",
    "segment_frames",
]

DEFAULT_LABELS = {
    2: ["background", "particle"],
    3: ["background", "intermediate", "bright"],
    4: ["background", "low", "intermediate", "bright"],
}


def _as_array(source):
    """Accept a HyperSpy signal or a plain array; return ``(frames, y, x)``."""
    data = getattr(source, "data", source)
    if isinstance(data, da.Array):
        data = data.compute()
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(
            f"the segmentation source must be (frames, y, x); got shape {data.shape}"
        )
    return data


def _overlay_coords(mask_shape, image_shape):
    """Pixel-centre coordinates that place a mask onto an image's pixel grid.

    ``imshow`` of an ``(h, w)`` array spans -0.5 to w-0.5 and -0.5 to h-0.5, so
    a mask pixel ``j`` reduced from the image by a factor ``f`` is centred at
    ``(j + 0.5) * f - 0.5``.  Without this, coarse masks drawn over a finer
    image collapse into its top-left corner, scaled down by the binning factor.
    Returns ``(xs, ys)``, which is the identity when the grids already match.
    """
    scale_y = image_shape[0] / mask_shape[0]
    scale_x = image_shape[1] / mask_shape[1]
    ys = (np.arange(mask_shape[0]) + 0.5) * scale_y - 0.5
    xs = (np.arange(mask_shape[1]) + 0.5) * scale_x - 0.5
    return xs, ys


def _frame_thresholds(frame, n_classes, method, quantiles):
    from skimage.filters import threshold_multiotsu

    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        raise ValueError("frame contains no finite values")
    if method == "multiotsu":
        return np.atleast_1d(threshold_multiotsu(finite, classes=n_classes))
    if method == "quantile":
        if quantiles is None:
            quantiles = np.linspace(0, 100, n_classes + 1)[1:-1]
        return np.percentile(finite, np.atleast_1d(quantiles))
    raise ValueError(f"unknown method {method!r}")


def _clean(labels, n_classes, min_size, fill_holes):
    """Tidy a label image, keeping the classes mutually exclusive.

    Classes are written back in ascending order so a brighter class always wins
    where they overlap -- which is what you want for a bright core sitting
    inside a dimmer rim: filling the rim's holes does not swallow the core.
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import remove_small_objects

    out = np.zeros_like(labels)
    for klass in range(1, n_classes):
        mask = labels == klass
        if fill_holes:
            mask = binary_fill_holes(mask)
        if min_size > 1:
            mask = remove_small_objects(mask, min_size=int(min_size))
        out[mask] = klass
    return out


@dataclass
class MaskStack:
    """Per-frame region masks, mutually exclusive within each frame.

    ``masks`` is boolean for masks cut at their own resolution, or float
    weights in [0, 1] summing to 1 across classes after :func:`rebin_masks`
    has reduced finer masks onto a coarser grid.  Everything here, and
    :func:`masked_spectra`, works with either.
    """

    masks: np.ndarray
    """``(frames, classes, y, x)``, boolean or fractional weights."""
    labels: list
    thresholds: np.ndarray
    """``(frames, classes - 1)`` -- the thresholds actually used."""
    source_title: str = ""
    nav_axes: list = field(default_factory=list, repr=False)

    @property
    def n_frames(self):
        return self.masks.shape[0]

    @property
    def n_classes(self):
        return self.masks.shape[1]

    @property
    def is_fractional(self):
        return self.masks.dtype.kind == "f"

    @property
    def areas(self):
        """``(frames, classes)`` pixel counts, or effective counts if fractional."""
        return self.masks.sum(axis=(2, 3))

    @property
    def area_fraction(self):
        return self.areas / float(np.prod(self.masks.shape[2:]))

    def label_image(self):
        """``(frames, y, x)`` integer labels, 0 = first class."""
        if self.is_fractional:
            return self.masks.argmax(axis=1).astype(np.uint8)
        out = np.zeros(self.masks.shape[:1] + self.masks.shape[2:], dtype=np.uint8)
        for klass in range(1, self.n_classes):
            out[self.masks[:, klass]] = klass
        return out

    def index(self, label):
        if label not in self.labels:
            raise KeyError(f"{label!r} not in {self.labels}")
        return self.labels.index(label)

    def summary(self):
        kind = "fractional weights" if self.is_fractional else "boolean"
        lines = [
            f"MaskStack: {self.n_frames} frames x {self.n_classes} classes "
            f"on {self.masks.shape[2]}x{self.masks.shape[3]} ({kind})"
        ]
        fraction = self.area_fraction
        for i, label in enumerate(self.labels):
            lines.append(
                f"  {label:<14} {100 * fraction[:, i].mean():5.1f}% of the field "
                f"(first {100 * fraction[0, i]:.1f}%, last {100 * fraction[-1, i]:.1f}%)"
            )
        return "\n".join(lines)

    def plot(self, source=None, frames=(0, -1), figsize=None, colors=None):
        """Overlay the mask outlines on an image.

        ``source`` may be at a different resolution from the masks -- the usual
        case being SI-grid masks from :func:`rebin_masks` drawn over the
        full-resolution HAADF they were cut from.  The contours are mapped onto
        the image's pixel grid, so no binning factor has to be applied by hand.
        """
        import matplotlib.pyplot as plt

        frame_list = [f % self.n_frames for f in frames]
        image = _as_array(source) if source is not None else self.label_image()
        if image.shape[0] != self.n_frames:
            raise ValueError(
                f"source has {image.shape[0]} frames but the masks have {self.n_frames}"
            )
        xs, ys = _overlay_coords(self.masks.shape[2:], image.shape[1:])
        colors = colors or ["tab:cyan", "tab:orange", "tab:red", "tab:green"]
        figsize = figsize or (3.1 * len(frame_list), 3.6)
        figure, axes = plt.subplots(
            1, len(frame_list), figsize=figsize, constrained_layout=True, squeeze=False
        )
        for axis, frame in zip(axes[0], frame_list):
            axis.imshow(image[frame], cmap="gray")
            for klass in range(1, self.n_classes):
                axis.contour(
                    xs,
                    ys,
                    self.masks[frame, klass].astype(float),
                    levels=[0.5],
                    colors=[colors[(klass - 1) % len(colors)]],
                    linewidths=0.9,
                )
            axis.set_title(f"frame {frame}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        handles = [
            plt.Line2D([], [], color=colors[(k - 1) % len(colors)], label=self.labels[k])
            for k in range(1, self.n_classes)
        ]
        axes[0][-1].legend(handles=handles, fontsize=7, loc="lower right")
        return figure

    def plot_areas(self, times=None, figsize=(6.5, 3.2)):
        """Region area against frame or time -- the in-situ trend."""
        import matplotlib.pyplot as plt

        x = np.arange(self.n_frames) if times is None else np.asarray(times)
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        fraction = self.area_fraction
        for klass in range(1, self.n_classes):
            axis.plot(x, 100 * fraction[:, klass], ".-", ms=4, label=self.labels[klass])
        axis.set_xlabel("frame" if times is None else "time (s)")
        axis.set_ylabel("area (% of field)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        return figure


def segment_frames(
    source,
    *,
    n_classes=3,
    labels=None,
    method="multiotsu",
    smooth=1.0,
    min_size=4,
    fill_holes=True,
    thresholds=None,
    quantiles=None,
    threshold_smoothing=None,
):
    """Threshold every frame of ``source`` independently into ``n_classes``.

    Parameters
    ----------
    source : signal or array, ``(frames, y, x)``
        What to segment on, already on the grid you want masks for -- the
        rebinned HAADF from :func:`~gatan_insitu.make_navigator`, an
        :func:`elemental_map`, or a :func:`composition_ratio`.  Read the module
        docstring before segmenting on raw HAADF intensity.
    n_classes : int
        Number of intensity classes, lowest first.  Class 0 is normally
        background and is kept so the masks tile the frame.
    labels : list of str, optional
        Names for the classes, lowest first, e.g.
        ``["background", "Ni-rich", "Au-rich"]``.
    method : {'multiotsu', 'quantile', 'manual'}
        ``'multiotsu'`` recomputes Otsu thresholds per frame, which absorbs
        drift in beam current or detector gain.  ``'manual'`` uses
        ``thresholds``.
    smooth : float
        Gaussian sigma (pixels) applied before thresholding only; the returned
        masks index the unsmoothed data.
    min_size, fill_holes :
        Morphological tidying per class.  ``min_size`` is in pixels.
    thresholds : array, optional
        With ``method='manual'``: either ``(n_classes - 1,)`` applied to every
        frame, or ``(frames, n_classes - 1)`` per frame.
    quantiles : sequence, optional
        With ``method='quantile'``: percentiles for the class edges.
    threshold_smoothing : int, optional
        Median-filter the per-frame thresholds over this many frames before
        applying them.  Steadies a jittery series at the cost of adapting more
        slowly; leave as ``None`` to use each frame's own thresholds.
    """
    from scipy.ndimage import gaussian_filter, median_filter

    data = _as_array(source)
    n_frames = data.shape[0]
    if labels is None:
        labels = DEFAULT_LABELS.get(n_classes, [f"class {i}" for i in range(n_classes)])
    labels = list(labels)
    if len(labels) != n_classes:
        raise ValueError(f"{len(labels)} labels for {n_classes} classes")

    smoothed = np.stack(
        [gaussian_filter(frame, smooth) if smooth else frame for frame in data]
    )

    if method == "manual":
        if thresholds is None:
            raise ValueError("method='manual' needs thresholds=")
        per_frame = np.asarray(thresholds, dtype=float)
        if per_frame.ndim == 1:
            per_frame = np.tile(per_frame, (n_frames, 1))
    else:
        per_frame = np.stack(
            [
                _frame_thresholds(smoothed[f], n_classes, method, quantiles)
                for f in range(n_frames)
            ]
        )
    if per_frame.shape != (n_frames, n_classes - 1):
        raise ValueError(
            f"thresholds have shape {per_frame.shape}, "
            f"expected {(n_frames, n_classes - 1)}"
        )

    if threshold_smoothing:
        width = int(threshold_smoothing)
        per_frame = np.stack(
            [median_filter(per_frame[:, c], size=width, mode="nearest")
             for c in range(per_frame.shape[1])],
            axis=1,
        )

    masks = np.zeros((n_frames, n_classes) + data.shape[1:], dtype=bool)
    for frame in range(n_frames):
        raw_labels = np.digitize(smoothed[frame], per_frame[frame]).astype(np.uint8)
        cleaned = _clean(raw_labels, n_classes, min_size, fill_holes)
        for klass in range(n_classes):
            masks[frame, klass] = cleaned == klass

    nav_axes = []
    if hasattr(source, "axes_manager"):
        nav_axes = [
            {"name": a.name, "units": a.units, "scale": getattr(a, "scale", 1.0)}
            for a in source.axes_manager._axes
        ]

    stack = MaskStack(
        masks=masks,
        labels=labels,
        thresholds=per_frame,
        source_title=getattr(getattr(source, "metadata", None), "General", None)
        and source.metadata.General.title
        or "",
        nav_axes=nav_axes,
    )
    _logger.info(
        "segmented %d frames into %s; mean area %%: %s",
        n_frames,
        labels,
        np.round(100 * stack.area_fraction.mean(axis=0), 1).tolist(),
    )
    return stack


def rebin_masks(mask_stack, *, factor=None, target_shape=None, mode="fraction"):
    """Reduce masks cut at HAADF resolution onto the coarser SI grid.

    The HAADF of an SI is recorded at an integer multiple of the spectrum-image
    sampling (``SI.Scale``; 4x here, 472x444 against 118x111), so segmenting
    the **full-resolution** frames and reducing afterwards gives sharper, less
    smeared region boundaries than segmenting a rebinned image would.

    Parameters
    ----------
    factor : int or (int, int), optional
        Block size to reduce by.  Give this or ``target_shape``.
    target_shape : (int, int), optional
        The ``(y, x)`` grid to land on, e.g. ``signal.data.shape[1:3]``.
    mode : {'fraction', 'majority'}
        ``'fraction'`` (default) keeps the **fraction** of each block belonging
        to each class, so an SI pixel straddling a boundary contributes to both
        regions in proportion -- which is what it physically did, since the
        probe covered both.  :func:`masked_spectra` weights by these directly.
        ``'majority'`` assigns each block wholly to its commonest class, giving
        boolean masks again.

    Returns
    -------
    MaskStack on the coarser grid.
    """
    if (factor is None) == (target_shape is None):
        raise ValueError("give exactly one of factor= or target_shape=")
    if mode not in ("fraction", "majority"):
        raise ValueError(f"mode must be 'fraction' or 'majority', got {mode!r}")

    fine = mask_stack.masks
    fine_shape = fine.shape[2:]
    if target_shape is not None:
        target_shape = tuple(int(s) for s in target_shape)
        factors = []
        for got, want in zip(fine_shape, target_shape):
            ratio, remainder = divmod(got, want)
            if remainder or ratio < 1:
                raise ValueError(
                    f"mask grid {fine_shape} is not an integer multiple of the "
                    f"target {target_shape}"
                )
            factors.append(ratio)
    else:
        factors = [int(factor)] * 2 if np.isscalar(factor) else [int(f) for f in factor]
        for got, step in zip(fine_shape, factors):
            if got % step:
                raise ValueError(f"{got} is not divisible by {step}")

    n_frames, n_classes = fine.shape[:2]
    coarse_y = fine_shape[0] // factors[0]
    coarse_x = fine_shape[1] // factors[1]
    blocks = fine.astype(np.float32).reshape(
        n_frames, n_classes, coarse_y, factors[0], coarse_x, factors[1]
    )
    weights = blocks.mean(axis=(3, 5))

    if mode == "majority":
        winner = weights.argmax(axis=1)
        weights = np.zeros_like(weights, dtype=bool)
        for klass in range(n_classes):
            weights[:, klass] = winner == klass

    reduced = MaskStack(
        masks=weights,
        labels=list(mask_stack.labels),
        thresholds=mask_stack.thresholds.copy(),
        source_title=mask_stack.source_title,
        nav_axes=list(mask_stack.nav_axes),
    )
    _logger.info(
        "rebinned masks %s -> %s by %s (%s)",
        fine_shape,
        (coarse_y, coarse_x),
        tuple(factors),
        mode,
    )
    return reduced


def erode_masks(mask_stack, iterations=1, classes=None):
    """Shrink masks away from their boundaries.

    The control for "is this signal really from this region?".  A probe sitting
    just inside a boundary still collects from the far side, so a region's
    spectrum can be contaminated by its neighbour.  Eroding and re-measuring
    separates the two: a signal that *falls* as the mask shrinks was spillover;
    one that holds or rises is intrinsic.

    Erode before :func:`rebin_masks`, at the resolution the masks were cut at,
    so the step size is the fine pixel rather than the coarse one.

    Note this tests **lateral** mixing only. EELS integrates along the beam, so
    a region defined on a projected image can still contain a different phase
    above or below it in the same column; no amount of eroding will reveal that.

    Parameters
    ----------
    iterations : int
        Erosion steps, i.e. pixels removed from each boundary.
    classes : sequence of int or str, optional
        Which classes to erode; by default every class except the first
        (background). Pixels freed by the erosion become background.
    """
    from scipy.ndimage import binary_erosion

    if mask_stack.is_fractional:
        raise ValueError(
            "erode fractional masks before rebin_masks, while they are still "
            "boolean at the resolution they were cut at"
        )
    if classes is None:
        indices = list(range(1, mask_stack.n_classes))
    else:
        indices = [
            mask_stack.index(c) if isinstance(c, str) else int(c) for c in classes
        ]

    masks = mask_stack.masks.copy()
    for klass in indices:
        masks[:, klass] = np.stack([
            binary_erosion(frame, iterations=int(iterations))
            for frame in masks[:, klass]
        ])
    # Keep the classes tiling the frame: what was eroded away becomes class 0.
    masks[:, 0] = ~masks[:, 1:].any(axis=1)

    eroded = MaskStack(
        masks=masks,
        labels=list(mask_stack.labels),
        thresholds=mask_stack.thresholds.copy(),
        source_title=mask_stack.source_title,
        nav_axes=list(mask_stack.nav_axes),
    )
    _logger.info(
        "eroded %s by %d px: area %% %s -> %s",
        [mask_stack.labels[i] for i in indices],
        iterations,
        np.round(100 * mask_stack.area_fraction.mean(axis=0), 2).tolist(),
        np.round(100 * eroded.area_fraction.mean(axis=0), 2).tolist(),
    )
    return eroded


def masked_spectra(signal, mask_stack, *, mode="sum", dtype="float32"):
    """Sum (or average) ``signal`` inside each mask, frame by frame.

    One pass over the data: the contraction is done with ``da.einsum`` against
    the mask stack, so a 9 GB series is read once for all regions rather than
    once per region.

    Parameters
    ----------
    signal : lazy EELS signal, ``(frames, y, x | E)``
    mask_stack : MaskStack
        Must match the signal's navigation shape.
    mode : {'sum', 'mean'}
        ``'mean'`` divides by the pixel count of each region in each frame, so
        regions of different size are comparable.  Frames where a region is
        empty come back as NaN.

    Returns
    -------
    dict of ``{label: signal}``
        One spectrum series per region, each with the signal's ``Time``
        navigation axis and its energy axis.  Pixel counts are stored in
        ``metadata.Segmentation.pixels``.
    """
    if mode not in ("sum", "mean"):
        raise ValueError(f"mode must be 'sum' or 'mean', got {mode!r}")

    data = signal.data
    nav_dim = signal.axes_manager.navigation_dimension
    if nav_dim != 3:
        raise ValueError(
            f"expected a (frames, y, x | E) signal, got navigation dimension {nav_dim}"
        )
    nav_shape = data.shape[:3]
    if mask_stack.masks.shape[0] != nav_shape[0] or (
        mask_stack.masks.shape[2:] != nav_shape[1:]
    ):
        raise ValueError(
            f"masks are {mask_stack.masks.shape[0]} x {mask_stack.masks.shape[2:]} "
            f"but the signal navigation grid is {nav_shape}"
        )

    if not isinstance(data, da.Array):
        data = da.from_array(data, chunks=data.shape)
    weights = da.from_array(
        mask_stack.masks.astype(dtype),
        chunks=(data.chunks[0], mask_stack.n_classes) + mask_stack.masks.shape[2:],
    )
    # (frames, y, x, E) x (frames, class, y, x) -> (frames, class, E)
    totals = da.einsum("fyxe,fcyx->fce", data, weights)
    totals = np.asarray(totals.compute())

    counts = mask_stack.areas
    if mode == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            totals = totals / counts[:, :, None]
        totals[counts == 0] = np.nan

    import hyperspy.api as hs

    energy = signal.axes_manager.signal_axes[0]
    time = signal.axes_manager._axes[0]
    signal_type = signal.metadata.get_item("Signal.signal_type", "")
    out = {}
    for index, label in enumerate(mask_stack.labels):
        # Build a non-lazy signal: the result is a few hundred kB, and using
        # signal.__class__ would label a computed array as Lazy.
        series = hs.signals.Signal1D(
            totals[:, index, :],
            axes=[
                {
                    "name": time.name,
                    "size": totals.shape[0],
                    "scale": getattr(time, "scale", 1.0),
                    "offset": getattr(time, "offset", 0.0),
                    "units": time.units,
                    "navigate": True,
                },
                {
                    "name": energy.name,
                    "size": energy.size,
                    "scale": energy.scale,
                    "offset": energy.offset,
                    "units": energy.units,
                    "navigate": False,
                },
            ],
        )
        if signal_type:
            series.set_signal_type(signal_type)
        series.metadata.General.title = f"{label} ({mode})"
        series.metadata.set_item("Segmentation.label", label)
        series.metadata.set_item("Segmentation.mode", mode)
        series.metadata.set_item("Segmentation.pixels", counts[:, index])
        for key in ("Acquisition_instrument", "Signal"):
            if signal.metadata.has_item(key):
                series.metadata.set_item(key, signal.metadata.get_item(key).as_dictionary())
        out[label] = series
    return out


def elemental_map(signal, window, background=None, *, fast=True, show_progressbar=False):
    """Integrate an energy window per probe position, for every frame.

    Returns a computed ``(frames, y, x)`` signal suitable for
    :func:`segment_frames`.  With ``background=(start, stop)`` a power law is
    fitted over that range and removed first, which is what makes the result an
    elemental map rather than a slab of raw counts.

    Power-law fitting on dark-corrected data containing zeros and negatives
    emits numpy overflow/invalid warnings; they are harmless but noisy.
    """
    energy_name = signal.axes_manager.signal_axes[0].name
    if background is None:
        window_signal = signal.isig[float(window[0]) : float(window[1])]
    else:
        window_signal = signal.isig[float(background[0]) : float(window[1])]
        window_signal = window_signal.remove_background(
            signal_range=(float(background[0]), float(background[1])),
            fast=fast,
            show_progressbar=show_progressbar,
        )
        window_signal = window_signal.isig[float(window[0]) : float(window[1])]
    out = window_signal.sum(axis=energy_name)
    if getattr(out, "_lazy", False):
        out.compute(show_progressbar=show_progressbar)
    out.metadata.General.title = (
        f"{window[0]:g}-{window[1]:g} eV" + (" (bg removed)" if background else "")
    )
    return out


def jump_ratio_map(signal, pre, post, *, smooth=0.0, minimum=None, show_progressbar=False):
    """Post-edge / pre-edge intensity ratio per pixel, for every frame.

    A composition proxy that is insensitive to thickness to first order: both
    windows scale with thickness, so the ratio does not.  It needs no
    background model, which matters here -- power-law fitting on dark-corrected
    data with zeros and negatives is unstable and noisy.

    Parameters
    ----------
    pre, post : (float, float)
        Energy windows just below and just above the edge, in eV.  For Ni
        L2,3 at 855 eV: ``pre=(820, 845)``, ``post=(860, 885)``.
    smooth : float
        Gaussian sigma (pixels) applied to the two window maps *before*
        dividing.  Single-frame per-pixel EELS counts here are low enough that
        the unsmoothed ratio is mostly shot noise, so a sigma of 1.5-2 is
        usually needed to see anything; smoothing before the division is what
        keeps the ratio unbiased.
    minimum : float, optional
        Pixels whose pre-edge mean falls below this are set to 0 rather than
        dividing by noise.  A sensible value is a small fraction of the median
        pre-edge signal.

    Returns
    -------
    ``(frames, y, x)`` array, ready for :func:`segment_frames`.

    Notes
    -----
    Whether this is usable *as a segmentation source* depends entirely on
    counts.  On the 1 ms/pixel dual-EELS series this package was built against
    it is not: the Ni L ratio separates the regions by only ~0.13 while the
    within-region scatter is ~0.25 even at ``smooth=3``, so single-frame masks
    cut from it wander into vacuum.  Averaged over a region of a few thousand
    pixels the same ratio is perfectly clean (1.07 against 0.95), which is why
    it works well as the *check* in :func:`compare_regions` while HAADF does
    the segmenting.  Measure the contrast-to-noise before trusting it::

        r = gi.jump_ratio_map(hl, pre=(820, 845), post=(860, 885), smooth=2)
        a, b = r[masks.masks[:, 1]], r[masks.masks[:, 2]]
        print(abs(a.mean() - b.mean()) / a.std())   # want >> 1
    """
    energy_name = signal.axes_manager.signal_axes[0].name
    before = signal.isig[float(pre[0]) : float(pre[1])].mean(axis=energy_name)
    after = signal.isig[float(post[0]) : float(post[1])].mean(axis=energy_name)
    if getattr(before, "_lazy", False):
        before.compute(show_progressbar=show_progressbar)
    if getattr(after, "_lazy", False):
        after.compute(show_progressbar=show_progressbar)

    low = np.asarray(before.data, dtype=np.float64)
    high = np.asarray(after.data, dtype=np.float64)
    if smooth:
        from scipy.ndimage import gaussian_filter

        low = np.stack([gaussian_filter(frame, smooth) for frame in low])
        high = np.stack([gaussian_filter(frame, smooth) for frame in high])
    if minimum is None:
        positive = low[low > 0]
        minimum = 0.05 * np.median(positive) if positive.size else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(low > minimum, high / low, 0.0)
    return np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)


def compare_regions(spectra, reference, other, *, pre_window, edges=None, plot=True):
    """Is ``other`` a different material from ``reference``, or just thicker?

    Scales ``reference``'s mean spectrum onto ``other``'s over ``pre_window``
    -- a region with no edges, where the signal is pure background and so
    proportional to thickness -- then reports what is left.

    A flat residual means the two regions are the same material at different
    thicknesses, and the mask separated mass-thickness rather than chemistry.
    Edges standing out of the residual mean the regions really do differ.

    Parameters
    ----------
    spectra : dict
        Output of :func:`masked_spectra`, ideally with ``mode='mean'``.
    reference, other : str
        Keys into ``spectra``.
    pre_window : (float, float)
        An edge-free energy window used to match the two backgrounds.
    edges : dict, optional
        ``{name: (pre, post)}`` energy windows; the jump ratio of each is
        reported per region, which is thickness-free and so directly
        comparable.

    Returns
    -------
    dict with ``energy``, ``scale``, ``residual``, ``flatness`` (peak residual
    as a fraction of the scaled spectrum -- small means pure thickness) and
    ``jump_ratios``, plus ``figure`` when ``plot``.
    """
    axis = spectra[reference].axes_manager.signal_axes[0].axis
    mean = {k: np.nanmean(v.data, axis=0) for k, v in spectra.items()}
    a, b = mean[reference], mean[other]

    window = (axis > pre_window[0]) & (axis < pre_window[1])
    if not window.any():
        raise ValueError(f"pre_window {pre_window} selects no channels")
    scale = float(b[window].sum() / a[window].sum())
    residual = b - scale * a
    flatness = float(np.abs(residual).max() / np.abs(b).max())

    jumps = {}
    if edges:
        for name, (pre, post) in edges.items():
            lo = (axis > pre[0]) & (axis < pre[1])
            hi = (axis > post[0]) & (axis < post[1])
            jumps[name] = {
                key: float(values[hi].mean() / values[lo].mean())
                for key, values in mean.items()
            }

    result = {
        "energy": axis,
        "scale": scale,
        "residual": residual,
        "flatness": flatness,
        "jump_ratios": jumps,
    }

    if plot:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
        axes[0].plot(axis, scale * a, lw=0.8, label=f"{reference} x {scale:.2f}")
        axes[0].plot(axis, b, lw=0.8, label=other)
        axes[0].set_title(
            f"matched on the {pre_window[0]:g}-{pre_window[1]:g} eV background",
            fontsize=9,
        )
        axes[0].legend(fontsize=8)
        axes[1].plot(axis, residual, lw=0.8, color="k")
        axes[1].axhline(0, color="r", lw=0.6)
        for name, (pre, post) in (edges or {}).items():
            centre = 0.5 * (pre[1] + post[0])
            axes[1].axvline(centre, color="C0", lw=0.6, ls=":")
            axes[1].annotate(
                name, (centre, axes[1].get_ylim()[1]), fontsize=7,
                rotation=90, va="top", ha="right",
            )
        axes[1].set_title(
            f"residual: {other} - scaled {reference}  "
            f"(peak {100 * flatness:.0f}% of signal)",
            fontsize=9,
        )
        for ax in axes:
            ax.set_xlabel("energy loss (eV)")
        axes[0].set_ylabel("counts / pixel")
        result["figure"] = figure
    return result


def composition_ratio(numerator, denominator, *, floor=None):
    """``numerator / (numerator + denominator)`` -- a thickness-insensitive map.

    Both arguments are ``(frames, y, x)`` maps from :func:`elemental_map`.  The
    ratio cancels thickness to first order, so thresholding it separates
    composition rather than mass-thickness.  ``floor`` blanks pixels whose
    total signal falls below that value (vacuum, where the ratio is noise).
    """
    a = _as_array(numerator).astype(np.float64)
    b = _as_array(denominator).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"maps differ in shape: {a.shape} vs {b.shape}")
    total = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total > 0, a / total, 0.0)
    if floor is not None:
        ratio[total < floor] = 0.0
    return ratio

