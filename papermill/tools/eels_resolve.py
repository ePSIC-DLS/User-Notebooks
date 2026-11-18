
visit, sample = str(visit).strip(), str(sample).strip()
if not (visit and sample):
    raise ValueError('Set visit and sample')
template_notebook = template_notebook or os.path.join(visit, 'processing', 'EELS_insitu_processing.ipynb')
code_path = code_path or os.path.join(visit, 'processing', 'eels_batches')
data_root = data_root or os.path.join(visit, 'raw', sample)
output_root = output_root or os.path.join(visit, 'processing', sample)

# code_dir is the gatan_insitu package folder; the template wants the folder that *contains* it
code_dir = os.path.normpath(str(code_dir).strip())
if not os.path.isfile(os.path.join(code_dir, '__init__.py')):
    raise FileNotFoundError(f'code_dir={code_dir!r} is not the gatan_insitu package folder '
                            '(no __init__.py there): copy workflows_eels/gatan_insitu and point code_dir at it')
if os.path.basename(code_dir) != 'gatan_insitu':
    raise ValueError(f'code_dir must end in gatan_insitu (the package name), got {code_dir!r}')
common_parameters = {**common_parameters, 'gatan_insitu_path': os.path.dirname(code_dir)}

for name in ('template_notebook', 'code_path', 'data_root', 'output_root', 'code_dir'):
    print(f'{name:18s} {globals()[name]}')
if not os.path.isfile(template_notebook):
    raise FileNotFoundError(f'{template_notebook} not found: copy papermill/EELS_insitu_processing.ipynb there, '
                            'or set template_notebook')
if not os.path.isdir(data_root):
    raise FileNotFoundError(f'{data_root} not found: check visit and sample')
