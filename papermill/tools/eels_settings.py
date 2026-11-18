
# ============================================================================
#  A.  WHERE THE DATA IS
# ============================================================================
# every folder in <visit>/raw/<sample> holding EELS data is a dataset,
# e.g. /dls/e01/data/2026/mg42736-3/raw/AuNi_GF/InSitu (7)
visit = '/dls/e01/data/2026/mg42736-3'
sample = 'AuNi_GF'

# the gatan_insitu package folder (the one holding __init__.py), e.g. a copy of workflows_eels/gatan_insitu
code_dir = '/dls/e01/data/2026/mg42736-3/processing/code/gatan_insitu'

# --- template and where the batch files go ('' = the default in brackets)
template_notebook = ''  # [<visit>/processing/EELS_insitu_processing.ipynb]
code_path = ''          # [<visit>/processing/eels_batches]


# ============================================================================
#  B.  WHAT THE EELS NOTEBOOK DOES  --  these are the template's own parameters
# ============================================================================
# Everything in this block is passed straight to EELS_insitu_processing.ipynb.
# Anything left out keeps the template's default; the check cell below prints
# every parameter the template accepts, so nothing here has to be guessed.
# `raw_data_path` and `outpath` are NOT set here: this notebook fills them in
# per dataset.
common_parameters = {
    # --- alignment
    'align_zlp': True,        # align the low loss on its zero-loss peak and apply the same
                              # per-pixel shifts to the high loss. Fixes the absolute energy
                              # scale; reads both streams twice (~15-25 min for 50 frames)
    'align_frames': False,    # register the frames for specimen drift, measured on the image
                              # series and applied to every stream
    'frame_reference': 'previous',   # 'previous' (robust to a changing specimen) or 'first'

    # --- which frames to process
    # The GIF is written from every loaded frame with the numbers burnt in, so the usual
    # routine is: run once with the defaults, watch the GIF, then set these and run again.
    'frame_start': 0,         # first frame to keep
    'frame_stop': 0,          # one past the last; 0 = to the end, negatives count back from it
    'frames': 0,              # load only the first N frames (a quick test); 0 = all

    # --- which image series to use for the GIF, the drift and the navigator
    'image_stream': '',       # '' = whichever detector image the dataset has (HAADF, ADF,
                              # DF ...), picked by gi.image_streams. Name one to override

    # --- outputs
    'make_gif': True,         # animated GIF of the image series, frame numbers burnt in
    'gif_fps': 5,             # frames per second in that GIF
    'make_quicklook': True,   # re-read the saved files and draw an overview figure
    'compression': 'gzip',    # 'gzip' (about half the size) or 'none' (faster)
}

# per-dataset overrides, keyed by dataset folder name. This is where a dataset with beam
# damage gets its own frame range, e.g.
#   {'InSitu (7)': {'frame_start': 2, 'frame_stop': 38},
#    'InSitu (9)': {'align_frames': True}}
per_dataset_parameters = {}


# ============================================================================
#  C.  WHICH FOLDERS COUNT AS DATASETS   (batch plumbing, not notebook parameters)
# ============================================================================
data_root = ''          # [<visit>/raw/<sample>]
data_glob = '*/*EELS*.dm4'  # a folder with an EELS .dm4 in it is a dataset
exclude = []
dataset_is = 'parent'   # raw_data_path gets the dataset folder, not the .dm4
dataset_id = 'name'     # datasets are named by their folder, e.g. 'InSitu (7)'
output_root = ''        # [<visit>/processing/<sample>]: results in <visit>/processing/<sample>/<dataset>
result_folder_name = ''
done_file = 'processing_summary.json'  # written last by the template: the dataset is finished
skip_done = True        # skip datasets that are finished


# ============================================================================
#  D.  SLURM
# ============================================================================
ssh_host = 'wilson'  # used only if sbatch is not available where this notebook runs
partition = 'cs04r'
time_limit = '06:00:00'
mem = '60G'
cpus_per_task = 4
concurrent_jobs = 5
env_setup = 'module load python/epsic3.13'
kernel_name = 'python3'
cell_timeout = 0  # seconds allowed per cell (0: no limit)

submit = False  # set True to submit; False writes everything but does not submit

# to follow / check a batch written earlier (e.g. after a kernel restart), give its batch id here,
# e.g. '20261016_144039'. Nothing new is written or submitted.
check_batch = ''
