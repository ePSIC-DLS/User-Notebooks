"""Lazy loading of Gatan multiframe (in-situ) STEM/EELS records into HyperSpy.

    import gatan_insitu as gi

    print(gi.describe_folder(folder))
    signals = gi.load_insitu_dataset(folder)
    hl = signals["EELS HL"]          # LazyEELSSpectrum, (time, y, x | E)
"""

from .alignment import align_frames, frame_drift
from .dm_tags import (
    DM_IMAGE_DTYPES,
    RawFormatInfo,
    find_raw_files,
    read_dm_tags,
    read_raw_format_info,
)
from .loader import (
    describe_folder,
    load_insitu_dataset,
    load_insitu_signal,
    save_rechunked,
)
from .notebook import computed, make_navigator, quicklook, summary_table
from .raw_store import RawFrameStore
from .segmentation import (
    MaskStack,
    compare_regions,
    composition_ratio,
    elemental_map,
    erode_masks,
    jump_ratio_map,
    masked_spectra,
    rebin_masks,
    segment_frames,
)
from .spectroscopy import (
    align_to_zlp,
    zlp_position,
    fit_white_lines,
    hyperspy_white_lines,
    fixed_pattern,
    remove_fixed_pattern,
    rolling_frames,
    bin_frames,
    edge_intensity,
    edge_table,
    plot_edge_series,
    power_law_background,
    trend,
    white_line_ratio,
)

__all__ = [
    "DM_IMAGE_DTYPES",
    "MaskStack",
    "RawFormatInfo",
    "RawFrameStore",
    "compare_regions",
    "bin_frames",
    "align_frames",
    "align_to_zlp",
    "composition_ratio",
    "computed",
    "describe_folder",
    "edge_intensity",
    "edge_table",
    "elemental_map",
    "erode_masks",
    "jump_ratio_map",
    "find_raw_files",
    "fit_white_lines",
    "fixed_pattern",
    "frame_drift",
    "hyperspy_white_lines",
    "load_insitu_dataset",
    "load_insitu_signal",
    "make_navigator",
    "masked_spectra",
    "plot_edge_series",
    "power_law_background",
    "quicklook",
    "rebin_masks",
    "read_dm_tags",
    "remove_fixed_pattern",
    "rolling_frames",
    "read_raw_format_info",
    "save_rechunked",
    "segment_frames",
    "summary_table",
    "trend",
    "white_line_ratio",
    "zlp_position",
]

__version__ = "0.1.0"





