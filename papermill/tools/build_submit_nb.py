"""Build Submit_papermill_array.ipynb: batch submission of a papermill notebook as a SLURM array."""
import sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []


def new_batch_only(src, when_checking):
    # wrap a cell so it only runs when writing a new batch, not when checking an existing one
    body = "\n".join(("    " + l) if l.strip() else l for l in src.strip("\n").splitlines())
    return f"if check_batch:\n    {when_checking}\nelse:\n{body}\n"

md = lambda s: cells.append(new_markdown_cell(s.strip("\n")))
code = lambda s, tags=None: cells.append(
    new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {})
)

md(r"""
# Run a papermill notebook over many datasets as a SLURM array job

In this notebook one can:
- check the batch settings against the parameters the template notebook accepts
- find the datasets to process (skipping those already finished)
- write one parameter file per dataset and a SLURM array script
- submit it, then follow the jobs and collect which datasets succeeded or failed

Each array task runs

    python -m papermill <template snapshot> <results folder>/<template name>_<dataset id>.ipynb -f <params>.json

It works with any template that takes `raw_data_path` and `outpath` parameters and writes a
file when it has finished (`done_file`): presets for the azimuthal variance and the Gatan
multiframe EELS notebooks are in the settings cell.

(the output name and folder are set in the settings), so every executed notebook keeps its own parameters, and a failing notebook makes its SLURM task fail.

Everything for one batch (template snapshot, parameter files, job list, script, logs) goes in
`<code_path>/<batch id>`, so later edits to the template do not change jobs already queued.

Run the cells top to bottom. Nothing is submitted until `submit = True`.

To come back to a batch later (after a kernel restart, say), set `check_batch` to its batch id and run
from the top again: the batch is reloaded instead of a new one being written or submitted.
""")

md("## Settings")

code(r"""
# --- template and where the batch files go
template_notebook = '/dls/e02/data/2026/mg44469-1/processing/Azimuthal_variance.ipynb'
code_path = '/dls/e02/data/2026/mg44469-1/processing/variance_batches'

# --- datasets: every path matching data_root/data_glob, minus paths containing any of `exclude`
data_root = '/dls/e02/data/2026/mg44469-1/processing/Merlin/MIL101_pristine_powder-2'
data_glob = '*/*_data.hdf5'
exclude = ['bin']
dataset_is = 'match'      # what raw_data_path gets: 'match' (the matched path) or 'parent' (its folder)
dataset_id = 'timestamp'  # how a dataset is named: 'timestamp' (first 15 characters) or 'name'
# results go to <output_root>/<dataset folder relative to data_root>/<result_folder_name> if output_root
# is set, otherwise to <dataset folder>/<result_folder_name>
output_root = ''
result_folder_name = 'Variance_Analysis'
done_file = '{id}_summary.json'  # in the results folder; when it exists the dataset is finished
skip_done = True  # skip datasets that are finished

# Presets (copy over the block above):
#  Azimuthal variance (Merlin NBED):  template Azimuthal_variance.ipynb, data_glob '*/*_data.hdf5',
#      dataset_is 'match', dataset_id 'timestamp', output_root '', result_folder_name 'Variance_Analysis',
#      done_file '{id}_summary.json'
#  Gatan multiframe EELS:  template EELS_insitu_processing.ipynb, data_root <visit>/<...>/<sample>,
#      data_glob '*/*EELS*.dm4', dataset_is 'parent', dataset_id 'name',
#      output_root <visit>/processing/<...>/<sample>, result_folder_name '', done_file 'processing_summary.json'

# --- notebook parameters shared by all datasets (anything not set uses the template default)
common_parameters = {
    'calib_path': '/dls/e02/data/2026/mg44469-1/processing/Merlin/au_xgrating_40cm',
    'origin_method': 'bragg',
    'peak_finding_minAbsoluteIntensity': 0.4,
}
# per-dataset overrides, keyed by dataset id, e.g. {'20260909_131114': {'crop_q': 100}}
per_dataset_parameters = {}

# --- SLURM
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
""", tags=["parameters"])

code(r"""
import os
import csv
import glob
import json
import shutil
import difflib
import datetime
import subprocess

import nbformat
import papermill as pm

print('papermill', pm.__version__)

# papermill -p turns a batch id like 20261016_144039 into the int 20261016144039: put the _ back
check_batch = str(check_batch).strip() if check_batch else ''
if check_batch.isdigit() and len(check_batch) == 14:
    check_batch = check_batch[:8] + '_' + check_batch[8:]
""")

md("## Check the settings against the template's parameters\n\nCatches misspelt parameter names, which papermill would otherwise inject silently and the template would ignore.")

code(r"""
template_params = pm.inspect_notebook(template_notebook)
template_stem = os.path.splitext(os.path.basename(template_notebook))[0]
print(f'{len(template_params)} parameters in {template_notebook}:')
for name, p in template_params.items():
    print(f"  {name:36s} {p['default']}")

batch_controlled = {'raw_data_path', 'outpath'}  # set per dataset below
requested = set(common_parameters)
for overrides in per_dataset_parameters.values():
    requested |= set(overrides)

unknown = sorted(requested - set(template_params))
if unknown:
    hints = {k: difflib.get_close_matches(k, template_params, n=1) for k in unknown}
    raise KeyError('Not parameters of the template: '
                   + ', '.join(f"{k} (did you mean {h[0]}?)" if h else k for k, h in hints.items()))
clash = sorted(requested & batch_controlled)
if clash:
    raise KeyError(f'{clash} are set per dataset by this notebook; do not set them here')
print('\nsettings OK')
""")

md("## Find the datasets")

code(r"""
if dataset_is not in ('match', 'parent'):
    raise ValueError("dataset_is must be 'match' or 'parent'")
if dataset_id not in ('timestamp', 'name'):
    raise ValueError("dataset_id must be 'timestamp' or 'name'")

matches = (m for m in glob.glob(os.path.join(glob.escape(data_root), data_glob))
           if not any(e in m for e in exclude))
# with dataset_is = 'parent', several matches in one folder are one dataset
data_files = sorted({os.path.dirname(m) if dataset_is == 'parent' else m for m in matches})
print(f'{len(data_files)} datasets found')

def time_stamp_of(path):
    # the dataset id: e.g. '20260909_131114' (timestamp rule of the azimuthal template) or 'InSitu (5)'
    name = os.path.basename(os.path.normpath(path))
    return name[:15] if dataset_id == 'timestamp' else name

def outpath_of(path):
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if output_root:
        folder = os.path.join(output_root, os.path.relpath(folder, data_root))
    return os.path.join(folder, result_folder_name) if result_folder_name else folder

def summary_of(path):
    return os.path.join(outpath_of(path), done_file.format(id=time_stamp_of(path)))

def safe(name):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name).strip('_')

todo = [f for f in data_files if not (skip_done and os.path.exists(summary_of(f)))]
for f in data_files:
    print(('  todo  ' if f in todo else '  done  ') + f)

unused = sorted(set(per_dataset_parameters) - {time_stamp_of(f) for f in data_files})
if unused:
    print(f'\nWARNING: per_dataset_parameters for timestamps not found: {unused}')
if not todo:
    print('\nNothing to do.')
""")

md("## Write the batch: template snapshot, one parameter file per dataset, job list")

code(new_batch_only(r"""
batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
batch_dir = os.path.join(code_path, batch_id)
params_dir = os.path.join(batch_dir, 'params')
logs_dir = os.path.join(batch_dir, 'logs')
for d in (params_dir, logs_dir):
    os.makedirs(d, exist_ok=True)

template_snapshot = os.path.join(batch_dir, os.path.basename(template_notebook))
shutil.copy2(template_notebook, template_snapshot)

jobs = []
for i, f in enumerate(todo):
    ts = time_stamp_of(f)
    params = {**common_parameters, **per_dataset_parameters.get(ts, {}),
              'raw_data_path': f, 'outpath': outpath_of(f)}
    params_file = os.path.join(params_dir, f'{i:03d}_{safe(ts)}.json')  # JSON is valid YAML for papermill -f
    with open(params_file, 'w') as fp:
        json.dump(params, fp, indent=2)
    output_notebook = os.path.join(outpath_of(f), f'{template_stem}_{safe(ts)}.ipynb')
    jobs.append({'task': i, 'time_stamp': ts, 'raw_data_path': f, 'params_file': params_file,
                 'output_notebook': output_notebook, 'summary': summary_of(f)})

jobs_file = os.path.join(batch_dir, 'jobs.tsv')
with open(jobs_file, 'w') as fp:
    fp.write(''.join(f"{j['params_file']}\t{j['output_notebook']}\n" for j in jobs))

manifest_file = os.path.join(batch_dir, 'manifest.json')
manifest = {'batch_id': batch_id, 'template': template_notebook, 'template_snapshot': template_snapshot,
            'common_parameters': common_parameters, 'per_dataset_parameters': per_dataset_parameters,
            'jobs': jobs, 'slurm_job_id': None}
with open(manifest_file, 'w') as fp:
    json.dump(manifest, fp, indent=2)

print(f'batch {batch_id}: {len(jobs)} jobs written to {batch_dir}')
""", 'batch_dir = os.path.join(code_path, check_batch)\n    manifest_file = os.path.join(batch_dir, \'manifest.json\')\n    with open(manifest_file) as fp:\n        manifest = json.load(fp)\n    jobs = manifest[\'jobs\']\n    print(f"checking batch {check_batch}: {len(jobs)} jobs, SLURM job {manifest[\'slurm_job_id\']}")'))

md("## Write the SLURM array script")

code(r"""
timeout_flag = f' --execution-timeout {int(cell_timeout)}' if cell_timeout else ''
bash_script_path = os.path.join(batch_dir, 'cluster_submit.sh')
if check_batch:  # paths of the reloaded batch
    jobs_file = os.path.join(batch_dir, 'jobs.tsv')
    logs_dir = os.path.join(batch_dir, 'logs')
    template_snapshot = manifest['template_snapshot']

script = f'''#!/usr/bin/env bash
#SBATCH --partition {partition}
#SBATCH --job-name {template_stem[:40]}
#SBATCH --time {time_limit}
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {cpus_per_task}
#SBATCH --mem {mem}
#SBATCH --array=0-{max(len(jobs) - 1, 0)}%{int(concurrent_jobs)}
#SBATCH --output={logs_dir}/%A_%a.out
#SBATCH --error={logs_dir}/%A_%a.err

{env_setup}
set -eo pipefail
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

# line N+1 of the job list: <params file> TAB <output notebook>
IFS=$'\\t' read -r PARAMS OUTNB < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{jobs_file}")
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $PARAMS -> $OUTNB"
mkdir -p "$(dirname "$OUTNB")"

python -m papermill "{template_snapshot}" "$OUTNB" -f "$PARAMS" -k {kernel_name} \\
    --log-output --cwd "$(dirname "$OUTNB")"{timeout_flag}
'''
if check_batch:
    print(f'checking an existing batch: {bash_script_path} not rewritten')
else:
    with open(bash_script_path, 'w', newline='\n') as fp:
        fp.write(script)
    print(script)
""")

md("## Submit\n\nUses `sbatch` directly if it is available here, otherwise via `ssh` to `ssh_host`.")

code(r"""
def run_on_cluster(args, timeout=120):
    # run a SLURM command here if possible, otherwise on the submission host
    cmd = args if shutil.which(args[0]) else ['ssh', ssh_host] + args
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{res.stderr}")
    return res.stdout


if check_batch:
    print('checking an existing batch: nothing submitted')
elif not jobs:
    print('No jobs to submit.')
elif not submit:
    print(f'submit is False: nothing submitted. To submit by hand:\n  sbatch {bash_script_path}')
else:
    job_id = run_on_cluster(['sbatch', '--parsable', bash_script_path]).strip().split(';')[0]
    manifest['slurm_job_id'] = job_id
    with open(manifest_file, 'w') as fp:
        json.dump(manifest, fp, indent=2)
    print(f'Submitted array job {job_id} ({len(jobs)} tasks, {concurrent_jobs} at a time)')
""")

md("## Follow the jobs\n\nRe-run this cell to refresh.")

code(r"""
job_id = manifest.get('slurm_job_id')
if job_id:
    print(run_on_cluster(['sacct', '-j', job_id, '-X',
                          '--format=JobID%18,State%12,Elapsed,ExitCode,NodeList%20']))
else:
    print('Not submitted from this notebook.')
""")

md(r"""
## Check the results

For every dataset in the batch: **done** if its summary file exists, **failed** if papermill recorded an
exception (with the failing cell and error), **incomplete** if the notebook exists but neither (still running, or killed by SLURM, e.g. on time or memory), **not started** otherwise.

To re-run only the failed datasets once the array has finished: fix the cause, keep `skip_done = True` and run this notebook again from the top.
""")

code(r"""
def status_of(job):
    if os.path.exists(job['summary']):
        return 'done', ''
    if not os.path.exists(job['output_notebook']):
        return 'not started', ''
    nb = nbformat.read(job['output_notebook'], as_version=4)
    if not nb.metadata.get('papermill', {}).get('exception'):
        return 'incomplete', ''
    for i, cell in enumerate(nb.cells):
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                return 'failed', f"cell {cell.get('execution_count', i)}: {out['ename']}: {out['evalue'][:200]}"
    return 'failed', ''


counts = {}
for job in jobs:
    state, detail = status_of(job)
    job['state'] = state
    counts[state] = counts.get(state, 0) + 1
    print(f"{job['time_stamp']}  {state:12s} {detail}")
print('\n', counts)
""")

md("## Collect the results of all finished datasets into one table")

code(r"""
def flatten(d, prefix=''):
    # one level of columns per nested key; short lists become space-separated text, long ones are left out
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            yield from flatten(v, key + '.')
        elif isinstance(v, list):
            if len(v) <= 20 and not any(isinstance(x, (list, dict)) for x in v):
                yield key, ' '.join(f'{x:.6g}' if isinstance(x, float) else str(x) for x in v)
        else:
            yield key, v


rows = []
for f in data_files:
    if not os.path.exists(summary_of(f)) or not summary_of(f).endswith('.json'):
        continue
    with open(summary_of(f)) as fp:
        rows.append({'dataset': time_stamp_of(f), 'results': outpath_of(f), **dict(flatten(json.load(fp)))})

if rows:
    columns = list(dict.fromkeys(k for r in rows for k in r))  # union, in first-seen order
    table_path = os.path.join(output_root or data_root, f'{template_stem}_summary.csv')
    with open(table_path, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"  {r['dataset']:<24} {r['results']}")
    print(f'\n{len(rows)} finished datasets, {len(columns)} columns, written to {table_path}')
else:
    print('No finished datasets yet.')
""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbformat.validate(nb)
nbformat.write(nb, sys.argv[1])
print("wrote", sys.argv[1], len(cells), "cells")
