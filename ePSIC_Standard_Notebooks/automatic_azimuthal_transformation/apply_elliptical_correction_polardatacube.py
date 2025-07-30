import os
import sys
import glob
import json
import h5py
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

import py4DSTEM
from py4DSTEM import DataCube, show
from py4DSTEM.preprocess.preprocess import filter_hot_pixels, datacube_diffraction_shift
from py4DSTEM.process.calibration import fit_ellipse_amorphous_ring
from py4DSTEM.process.calibration.origin import get_probe_size, get_origin, fit_origin
from py4DSTEM.process.utils.elliptical_coords import cartesian_to_polarelliptical_transform
from py4DSTEM.visualize import show, add_scalebar, add_ellipses, add_circles
from py4DSTEM.process.polar.polar_datacube import PolarDatacube
from scipy.signal import find_peaks
from scipy.ndimage import rotate, maximum_filter1d, gaussian_filter1d
from skimage.transform import SimilarityTransform, AffineTransform, warp
from emdfile import tqdmnd


info_path = sys.argv[1]
index = int(sys.argv[2])
info = {}
with open(info_path, 'r') as f:
    for line in f:
        tmp = line.split(" ")
        if tmp[0] == 'data_labels':
            info[tmp[0]] = line.split(" = ")[1].split('\n')[:-1]
            print(tmp[0], line.split(" = ")[1].split('\n')[:-1])
        else:
            info[tmp[0]] = tmp[-1].split("\n")[0]
            print(tmp[0], tmp[-1].split("\n")[0])

# Assignment of global variables
# dataset name
DATA_LABEL = eval(info['data_labels'][0])[index]
basedir = info['basedir']
YEAR = info['YEAR']
VISIT = info['VISIT']
sub = info['sub']
also_rpl = eval(info['also_rpl'])

cal_timestamp = DATA_LABEL.split('/')[-1]
base_path = f'{basedir}/{YEAR}/{VISIT}/processing/Merlin'
meta_path = f'{base_path}/{DATA_LABEL}/{cal_timestamp}.hdf'
data_path = f'{base_path}/{DATA_LABEL}/{cal_timestamp}_data.hdf5'
print("data_path:"+data_path)
save_dir = f'{base_path}/{DATA_LABEL}'

with h5py.File(meta_path,'r') as f:
    print("----------------------------------------------------------")
    print(f['metadata']["step_size(m)"])
    print(f['metadata']["step_size(m)"][()])
    R_SCALE = f['metadata']["step_size(m)"][()]
    nominal_CL = f['metadata']['nominal_camera_length(m)'][()]
    print("----------------------------------------------------------")

json_calibration_files = glob.glob(info['au_calib_dir'] + '/**/*.json', recursive = True)
cal_dict_list = []
for this_cal_file in json_calibration_files:
    print(this_cal_file)
    with open(this_cal_file, 'r') as fp:
        cal_dict_list.append(json.load(fp))
print(*cal_dict_list, sep="\n")

for this_dict in cal_dict_list:
    if this_dict['nominal_camera_length(m)'] == nominal_CL:
        Q_SCALE = this_dict['reciprocal_space_pix(1/A)']
        PARAMS = this_dict['p_ellipse']

R_Q_ROTATION = eval(info['R_Q_ROTATION'])
also_rpl = eval(info['also_rpl'])
mask_path = info['mask_path']
fast_origin = eval(info['fast_origin'])
print(type(R_SCALE), R_SCALE)
print(type(Q_SCALE), Q_SCALE)
print(type(PARAMS), PARAMS)
print(type(R_Q_ROTATION), R_Q_ROTATION)
print(type(DATA_LABEL), DATA_LABEL)
print(type(basedir), basedir)
print(type(YEAR), YEAR)
print(type(VISIT), VISIT)
print(type(also_rpl), also_rpl)
print(type(mask_path), mask_path)
print(type(fast_origin), fast_origin)

def fast_find_origin(datacube, dp, pad=None):
    (qrad_init, qx0_init, qy0_init) = get_probe_size(dp.data)
    if pad is None:
        window = int(2*qrad_init)
    else:
        window = qrad_init + int(pad)
    (qx0r, qy0r) = int(qx0_init//1), int(qy0_init//1)
    (x_lower, x_upper) =  qx0r - window, qx0r + window
    (y_lower, y_upper) =  qy0r - window, qy0r + window
    d_cent = DataCube(datacube.data[:,:,x_lower:x_upper,y_lower:y_upper])
    (qx0, qy0, __) = get_origin(d_cent)
    qx0 += x_lower
    qy0 += y_lower
    qx0[np.where(np.isnan(qx0))] = qx0_init
    qy0[np.where(np.isnan(qy0))] = qy0_init
    return qx0, qy0


def apply_affine(data, p_ellipse, r_q):
    (x0, y0, a, b, theta) = p_ellipse
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    
    scaling = np.array([[1, 0],
                        [0, b/a]])
    
    shift_x = int(x0)
    shift_y = int(y0)

    r_q_rot = np.array([[np.cos(r_q), -np.sin(r_q)],
                        [np.sin(r_q), np.cos(r_q)]])
    distortion = np.dot(r_q_rot, np.dot(rotation.T,np.dot(scaling, rotation)))# np.dot(shearing, rotation)))
    
    affine = np.array([[distortion[0, 0], distortion[0, 1], 0.00],
                       [distortion[1, 0], distortion[1, 1], 0.00],
                       [0.00, 0.00, 1.00]])
    
    tf_shift = SimilarityTransform(translation=[-shift_y, -shift_x])
    tf_shift_inv = SimilarityTransform(translation=[shift_y, shift_x])
    
    tf_distortion = AffineTransform(matrix=affine)
    tf_correction = (tf_shift + (tf_distortion + tf_shift_inv)).inverse
    warped = warp(data, tf_correction, order=1, preserve_range=True)
    return warped


def elliptical_resample_datacube_with_RQ_rotation(datacube, p_ellipse, r_q_rotation):
    
    datacube.calibration.set_p_ellipse(p_ellipse)
    
    polardata = PolarDatacube(datacube,
                                qmin = 0,
                                qmax = None,
                                qstep = 1.0,
                                n_annular = 180,
                                qscale = 1.0)
    
    mean_arr, var_arr = polardata.calculate_FEM_local(use_median=False,
                                                      return_values=True)
    ## Set sampling conditions for radial statistics
    for rx, ry in tqdmnd(datacube.R_Nx, datacube.R_Ny):
        datacube.data[rx, ry] = apply_affine(datacube.data[rx,ry], p_ellipse, r_q_rotation)
        
    return mean_arr, var_arr, datacube.data #datacube not returned as modified in-place

def build_hs_3d(array, r_scale, q_scale, name, **kwargs):
    hs_3d = hs.signals.Signal1D(array, **kwargs)
    for i in range(2):
        hs_3d.axes_manager[i].scale = r_scale*1e9
        hs_3d.axes_manager[i].units = 'nm'
        hs_3d.axes_manager[i].name = 'Navigation'
    hs_3d.axes_manager[2].scale = q_scale
    hs_3d.axes_manager[2].units = '1/Å'
    hs_3d.axes_manager[2].name = name
    return hs_3d

                   
if mask_path != '':
    try:
        with h5py.File(info["mask_path"],'r') as f:
            mask = f['data']['mask'][()]
        mask = np.bool_(mask)
        mask = np.invert(mask)
        mask = mask.astype(np.unit8)

    except:
        with h5py.File(info["mask_path"],'r') as f:
            mask = f['root']['np.array']['data'][()]
        mask = np.bool_(mask)
        mask = np.invert(mask)
        mask = mask.astype(np.unit8)        
    
    print(type(mask))
    print(mask.dtype)

## Load data using Hyperspy
data_hs = hs.load(data_path, reader='hspy', lazy=False)

if mask_path != '':
    ## Move to py4DSTEM DataCube (note: not a deepcopy)
    data = DataCube(np.multiply(data_hs.data, mask).astype(np.unit8))
    
else:
    ## Move to py4DSTEM DataCube (note: not a deepcopy)
    data = DataCube(data_hs.data)
    ## Filter hot pixels in-place
    filter_hot_pixels(data, thresh=0.1)

    
##Get initial raw mean diffraction pattern
data.get_virtual_diffraction(method='max',
                             name='dp_max_raw',
                             shift_center=False)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.sqrt(data.tree('dp_max_raw').data), cmap='gray')
ax.axis('off')
fig.tight_layout()
fig.savefig(f'{save_dir}{os.sep}{cal_timestamp}_original_max_dp.png')

if fast_origin:
    ## Get origin shifts
    (qx0, qy0) = fast_find_origin(data, dp=data.tree('dp_max_raw').data)

    ## Fit origin shifts
    (qx0_fit, qy0_fit, qx0_res, qy0_res) = fit_origin((qx0, qy0))
else:
    pad = None
    (qrad_init, qx0_init, qy0_init) = get_probe_size(data.tree('dp_max_raw').data)
    print('the radius of the probe: ', qrad_init)
    print('initial qx0 and qy0: (%f, %f)'%(qx0_init,
                                           qy0_init))
    if pad is None:
        window = int(2*qrad_init)
    else:
        window = qrad_init + int(pad)
    print("the window size: ", window)
    (qx0r, qy0r) = int(qx0_init//1), int(qy0_init//1)
    (x_lower, x_upper) =  int(qx0r - window), int(qx0r + window)
    (y_lower, y_upper) =  int(qy0r - window), int(qy0r + window)
    d_cent = DataCube(data.data[:,:,x_lower:x_upper,y_lower:y_upper])
    print(d_cent)

    probe_semiangle = 4

    syn_probe_width='3'
    syn_probe_rad = int(probe_semiangle)
    syn_probe_width = float(syn_probe_width)

    syn_probe = py4DSTEM.braggvectors.probe.Probe.generate_synthetic_probe(syn_probe_rad, 
                                                                           syn_probe_width, 
                                                                           (d_cent.data.shape[-1], 
                                                                            d_cent.data.shape[-1]))


    syn_probe_kernel = syn_probe.get_kernel(
        mode = 'sigmoid',
        radii = (probe_semiangle * 1, probe_semiangle * 3.0),
        bilinear=True,
    )

    detect_params = {
        'corrPower': 1.0,
        'sigma': 0,
        'edgeBoundary': 2,
        'minRelativeIntensity': 0,
        'minAbsoluteIntensity': 0.25,
        'minPeakSpacing': 2,
        'subpixel' : 'poly',
        'upsample_factor': 2,
        'maxNumPeaks': 1000,
    }

    bragg_peaks = d_cent.find_Bragg_disks(
        template = syn_probe_kernel,
        **detect_params,
    )
    
    
    # Measure the origin
    center_guess = (qx0r, qy0r)
    # radial_range = (8,200)
    qx0_meas,qy0_meas,mask_meas = bragg_peaks.measure_origin(
        center_guess=center_guess,
        score_method='intensity',
        # findcenter='max',
    )
    
    
    # Fit a plane to the origins
    qx0_fit, qy0_fit, qx0_residuals, qy0_residuals = bragg_peaks.fit_origin(
        robust= True,
        robust_thresh= 1.2,
    )
    
    qx0_fit += x_lower
    qy0_fit += y_lower
    qx0_fit[np.where(np.isnan(qx0_fit))] = qx0_init
    qx0_fit[np.where(np.isnan(qy0_fit))] = qy0_init


## Embed calibrations
data.calibration.set_origin((qx0_fit, qy0_fit))

## Shift DataCube centres to mean position for common centre
datacube_diffraction_shift(data,
                           xshifts=-data.calibration.get_qx0shift(),
                           yshifts=-data.calibration.get_qy0shift(),
                           bilinear=True)

## Neutralise origin calibrations to avoid issues later
data.calibration.set_qx0(data.calibration.get_qx0_mean()
                         *np.ones_like(data.calibration.get_qx0()))

data.calibration.set_qy0(data.calibration.get_qy0_mean()
                         *np.ones_like(data.calibration.get_qy0()))

# data.get_virtual_diffraction(method='max', name='dp_max_centered')

azi_mean, azi_var, data_hs.data = elliptical_resample_datacube_with_RQ_rotation(data, PARAMS, np.radians(R_Q_ROTATION))

mean_sig = build_hs_3d(azi_mean.data, R_SCALE, Q_SCALE, 'Azimuthal Mean')
# max_sig = build_hs_3d(azi_max, R_SCALE, Q_SCALE, 'Azimuthal Maximum')
var_sig = build_hs_3d(azi_var.data, R_SCALE, Q_SCALE, 'Azimuthal Variance')

## Save signals in HSPY and RPL
mean_sig.save(f'{save_dir}{os.sep}{cal_timestamp}_azimuthal_mean', overwrite=True, extension='hspy')
if also_rpl:
    mean_sig.save(f'{save_dir}{os.sep}{cal_timestamp}_azimuthal_mean', overwrite=True, extension='rpl')

var_sig.save(f'{save_dir}{os.sep}{cal_timestamp}_azimuthal_var', overwrite=True, extension='hspy')
if also_rpl:
    var_sig.save(f'{save_dir}{os.sep}{cal_timestamp}_azimuthal_var', overwrite=True, extension='rpl')

## Save elliptically corrected 4D data
# data_hs was modified by the datacube processing so does not need copying
for i in range(2):
    data_hs.axes_manager[i].scale = R_SCALE*1e9
    data_hs.axes_manager[i].units = 'nm'
    data_hs.axes_manager[i].name = 'Real Space Distance'
    data_hs.axes_manager[i+2].scale = Q_SCALE
    data_hs.axes_manager[i+2].units = '1/Å'
    data_hs.axes_manager[i+2].name = 'Reciprocal Space Distance'

    
max_dp = data_hs.max(axis=(0, 1))
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(np.sqrt(max_dp.data.clip(min=0.0)), cmap='gray')
ax.axis('off')
fig.tight_layout()
fig.savefig(f'{save_dir}{os.sep}{cal_timestamp}_azimuthal_data_centre.png')

data_hs.save(f'{save_dir}{os.sep}{cal_timestamp}_corrected_scaled', overwrite=True, extension='hspy')