# Python script for electron ptychography
# Jinseok Ryu (jinseok.ryu@diamond.ac.uk)


import sys
import numpy as np
import h5py as h5
import hyperspy.api as hs
import cupy as cp
from cupy.fft.config import get_plan_cache


# load the information
info_file = sys.argv[1]
info = {}
with open(info_file, "r") as f:
    for line in f:
        tmp = line.split(" ")
        info[tmp[0]] = tmp[2].split("\n")[0]
        print(tmp[0], tmp[2].split("\n")[0])

if info["package"] == "abtem_latest":
    import abtem
    from abtem.core.energy import energy2wavelength
    from abtem.reconstruct import MixedStatePtychographicOperator, MultislicePtychographicOperator, RegularizedPtychographicOperator
    from abtem.measurements import DiffractionPatterns

elif info["package"] == "abtem_legacy":
    from abtem import GridScan, PixelatedDetector, Potential, Probe, show_atoms, SMatrix, AnnularDetector
    from abtem.detect import PixelatedDetector
    from abtem.reconstruct import MixedStatePtychographicOperator, MultislicePtychographicOperator, RegularizedPtychographicOperator
    from abtem.measure import Measurement, Calibration, bandlimit, center_of_mass
    from abtem.utils import energy2wavelength

elif info["package"] == "py4dstem":
    import py4DSTEM
    from py4DSTEM.process.phase.utils import get_array_module

else:
    print("Wrong package!")

def force_fill(img, col_ind, row_ind, num_neighbor=5):
    pad = int(num_neighbor/2)
    padded = np.zeros((img.shape[0]+pad*2, img.shape[1]+pad*2))
    padded[pad:-pad, pad:-pad] = img.copy()

    filled_img = img.copy()
    
    for col, row in zip(col_ind, row_ind):
        temp = padded[(col):(col+2*pad), (row):(row+2*pad)]
        if np.nonzero(temp)[0].size != 0:
            avg = (np.sum(temp)-img[col, row]) / np.nonzero(temp)[0].size
        else:
            avg = (np.sum(temp)-img[col, row])
        filled_img[col, row] = avg

    return filled_img


# load the data
if info["mask_path"] != '':
    try:
        with h5.File(info["mask_path"],'r') as f:
            mask = f['data']['mask'][()]
        mask = np.bool_(mask)
        mask = np.invert(mask)
        mask = mask.astype(np.float32)

    except:
        with h5.File(info["mask_path"],'r') as f:
            mask = f['root']['np.array']['data'][()]
        mask = np.bool_(mask)
        mask = np.invert(mask)
        mask = mask.astype(np.float32)        
    
    print(type(mask))
    print(mask.dtype)

try:
    f = h5.File(info["data_path"], 'r')
    print(f)
    try:
        original_stack = f['Experiments']['__unnamed__']['data'][()]
    except:
        original_stack = f['data']['frames'][()]
    f.close()
except:
    original_stack = hs.load(info["data_path"]).data

original_stack = original_stack.astype(np.float32)
print("data shape: ", original_stack.shape)
print("data datatype: ", original_stack.dtype)

being_processed = original_stack
bp_shape = being_processed.shape

# option - RoI
if eval(info["crop_R"]):
    top, bottom, left, right = eval(info["crop_R_region"])
    being_processed = original_stack[top:bottom, left:right]
    bp_shape = being_processed.shape
    print("R-cropped data shape: ", bp_shape)

if info["mask_path"] != '':
    being_processed = np.multiply(being_processed, mask)

if eval(info["crop_Q"]):
    top, bottom, left, right = eval(info["crop_Q_region"])
    being_processed = being_processed[:, :, top:bottom, left:right]
    bp_shape = being_processed.shape
    print("Q-cropped data shape: ", bp_shape)

# option - fill the empty pixels
if eval(info["fill"]):
    top, bottom, left, right = eval(info["fill_region"])
    zero_ind = np.where(mask[top:bottom, left:right]==0)
    for i in range(bp_shape[0]):
        for j in range(bp_shape[1]):
            being_processed[i, j][top:bottom, left:right] = force_fill(being_processed[i, j, top:bottom, left:right],zero_ind[0], zero_ind[1], num_neighbor=eval(info["num_neighbor"]))

# binning
if bp_shape[3] % eval(info["binsize"]) != 0:
    remove_ = int(bp_shape[3] % eval(info["binsize"]))
    being_processed = being_processed[:, :, :-remove_, :-remove_]
    bp_shape = being_processed.shape
print(bp_shape)

being_processed = being_processed.clip(min=0.0)
being_processed -= np.min(being_processed)
being_processed /= np.max(being_processed)
being_processed *= 64

print("minimum intensity: ", np.min(being_processed))
print("maximum intensity: ", np.max(being_processed))
print("mean intensity: ", np.mean(being_processed))
print("standard deviation: ", np.std(being_processed))


### Use 'py4DSTEM' ###
if info["package"] == "py4dstem":
    dataset = py4DSTEM.DataCube(data=being_processed)
    print("original dataset")
    print(dataset)
    
    del being_processed
    del original_stack
    
    if eval(info["binsize"]) > 1:
        dataset.bin_Q(eval(info["binsize"]))
        print("after binning")
        print(dataset)
    
    bp_shape = dataset.data.shape
    
    dataset.get_dp_mean()
    dataset.get_dp_max()
    
    probe_radius_pixels, probe_qx0, probe_qy0 = dataset.get_probe_size(thresh_lower=0.01, thresh_upper=0.92, N=100, plot=False)
    semiangle = eval(info["semiangle"])*1E3
    center_x, center_y = eval(info["center"])
    
    dataset.calibration._params['Q_pixel_size'] = eval(info["reciprocal_pixel_size"])*1E3
    dataset.calibration._params['Q_pixel_units'] = "mrad"
    dataset.calibration._params['R_pixel_size'] = eval(info["scan_step"])
    dataset.calibration._params['R_pixel_units'] = "A"
    dataset.calibration._params['qx0_mean'] = center_x
    dataset.calibration._params['qy0_mean'] = center_y
    
    print(dataset)
    print(dataset.calibration)
    print("Rx extent: ", dataset.R_Ny*dataset.R_pixel_size)
    print("Ry extent: ", dataset.R_Nx*dataset.R_pixel_size)
    
    # py4DSTEM ptycho - singlislice
    if info["ptycho_type"] == "singleslice":
        print("singleslice ptychography reconstruction process begins")
        ptycho = py4DSTEM.process.phase.SingleslicePtychography(
            datacube= dataset,
            device = "cpu",
            energy = eval(info["HT"]),
            semiangle_cutoff = semiangle,
            defocus = eval(info["defocus"]),
            object_type='potential',
        ).preprocess(
            plot_center_of_mass = False, 
            plot_rotation = False, 
            plot_probe_overlaps = False, 
            force_com_rotation = eval(info["rotation"]), 
            force_com_transpose = False,
            fit_function = "plane",
        )
        
        ptycho_recon = ptycho.reconstruct(
            reset=True,
            store_iterations=True,
            reconstruction_method=info["reconstruction_method"],
            reconstruction_parameter=eval(info["reconstruction_parameter"]),
            device=info["device"],
            num_iter=eval(info["num_iteration"]),
            normalization_min=eval(info["reconstruction_parameter"]),
            object_positivity=eval(info["object_positivity"]),
            tv_denoise=eval(info["tv_denoise"]),
            tv_denoise_weight=eval(info["tv_denoise_weights"]),
            tv_denoise_inner_iter=eval(info["tv_denoise_inner_iter"]),
            max_batch_size=eval(info["max_batch_size"])
        )
        
        print("reconstruction completed")
        print("reconstructed real space pixel size (x, y)")
        print(ptycho_recon.sampling)
        
        save_object = hs.signals.Signal2D(np.asarray(ptycho_recon.object))
        save_probe = hs.signals.Signal2D(np.asarray(ptycho_recon.probe))
    
    # py4DSTEM ptycho - mixed state
    elif info["ptycho_type"] == "mixed-state":
        print("mixed-state ptychography reconstruction process begins")
        ptycho = py4DSTEM.process.phase.MixedstatePtychography(
            datacube=dataset,
            verbose=True,
            energy=eval(info["HT"]),
            num_probes=eval(info["num_probe"]),
            semiangle_cutoff=semiangle,
            defocus=eval(info["defocus"]),
            device="cpu",
            object_type='potential'
        ).preprocess(
            plot_center_of_mass = False, 
            plot_rotation = False, 
            plot_probe_overlaps = False, 
            force_com_rotation = eval(info["rotation"]), 
            force_com_transpose = False,
            fit_function = "plane",
        )
        
        ptycho_recon = ptycho.reconstruct(
            reset=True,
            store_iterations=True,
            reconstruction_method=info["reconstruction_method"],
            reconstruction_parameter=eval(info["reconstruction_parameter"]),
            device=info["device"],
            num_iter=eval(info["num_iteration"]),
            normalization_min=eval(info["reconstruction_parameter"]),
            object_positivity=eval(info["object_positivity"]),
            tv_denoise=eval(info["tv_denoise"]),
            tv_denoise_weight=eval(info["tv_denoise_weights"]),
            tv_denoise_inner_iter=eval(info["tv_denoise_inner_iter"]),
            max_batch_size=eval(info["max_batch_size"])
        )
        
        print("reconstruction completed")
        print("reconstructed real space pixel size (x, y)")
        print(ptycho_recon.sampling)
        
        save_object = hs.signals.Signal2D(np.asarray(ptycho_recon.object))
        save_probe = hs.signals.Signal2D(np.asarray(ptycho_recon.probe))
    
    # py4DSTEM ptycho - multislice
    elif info["ptycho_type"] == "multislice":
        print("multislice ptychography reconstruction process begins")
        ptycho = py4DSTEM.process.phase.MultislicePtychography(
            datacube=dataset,
            num_slices=eval(info["num_slice"]),
            slice_thicknesses=eval(info["slice_thickness"]),
            verbose=True,
            energy=eval(info["HT"]),
            defocus=eval(info["defocus"]),
            semiangle_cutoff=semiangle,
            device="cpu",
            object_type='potential'
        ).preprocess(
            plot_center_of_mass = False, 
            plot_rotation = False, 
            plot_probe_overlaps = False, 
            force_com_rotation = eval(info["rotation"]), 
            force_com_transpose = False,
            fit_function = "plane",
        )
        
        ptycho_recon = ptycho.reconstruct(
            reset=True,
            store_iterations=True,
            reconstruction_method=info["reconstruction_method"],
            reconstruction_parameter=eval(info["reconstruction_parameter"]),
            device=info["device"],
            num_iter=eval(info["num_iteration"]),
            normalization_min=eval(info["reconstruction_parameter"]),
            identical_slices=eval(info["identical_slices"]),
            object_positivity=eval(info["object_positivity"]),
            tv_denoise=eval(info["tv_denoise"]),
            tv_denoise_weights=eval(info["tv_denoise_weights"]),
            tv_denoise_inner_iter=eval(info["tv_denoise_inner_iter"]),
            tv_denoise_chambolle=eval(info["tv_denoise_chambolle"]),
            tv_denoise_weight_chambolle=eval(info["tv_denoise_weight_chambolle"]),
            max_batch_size=eval(info["max_batch_size"])
        )
        
        print("reconstruction completed")
        print("reconstructed real space pixel size (x, y)")
        print(ptycho_recon.sampling)
        
        save_object = hs.signals.Signal2D(np.asarray(ptycho_recon.object))
        save_probe = hs.signals.Signal2D(np.asarray(ptycho_recon.probe))

    # mixed-state multislice ptychography
    elif info["ptycho_type"] == "mixed-state-multislice":
        print("mixed-state multislice ptychography reconstruction process begins")
        ptycho = py4DSTEM.process.phase.MixedstateMultislicePtychography(
            datacube=dataset,
            num_probes=eval(info["num_probe"]),
            num_slices=eval(info["num_slice"]),
            slice_thicknesses=eval(info["slice_thickness"]),
            verbose=True,
            energy=eval(info["HT"]),
            defocus=eval(info["defocus"]),
            semiangle_cutoff=semiangle,
            device="cpu",
            object_type='potential'
        ).preprocess(
            plot_center_of_mass = False, 
            plot_rotation = False, 
            plot_probe_overlaps = False, 
            force_com_rotation = eval(info["rotation"]), 
            force_com_transpose = False,
            fit_function = "plane",
        )
        
        ptycho_recon = ptycho.reconstruct(
            reset=True,
            store_iterations=True,
            reconstruction_method=info["reconstruction_method"],
            reconstruction_parameter=eval(info["reconstruction_parameter"]),
            device=info["device"],
            num_iter=eval(info["num_iteration"]),
            normalization_min=eval(info["reconstruction_parameter"]),
            identical_slices=eval(info["identical_slices"]),
            object_positivity=eval(info["object_positivity"]),
            tv_denoise=eval(info["tv_denoise"]),
            tv_denoise_weights=eval(info["tv_denoise_weights"]),
            tv_denoise_inner_iter=eval(info["tv_denoise_inner_iter"]),
            tv_denoise_chambolle=eval(info["tv_denoise_chambolle"]),
            tv_denoise_weight_chambolle=eval(info["tv_denoise_weight_chambolle"]),
            max_batch_size=eval(info["max_batch_size"])
        )
        
        print("reconstruction completed")
        print("reconstructed real space pixel size (x, y)")
        print(ptycho_recon.sampling)
        
        save_object = hs.signals.Signal2D(np.asarray(ptycho_recon.object))
        save_probe = hs.signals.Signal2D(np.asarray(ptycho_recon.probe))


    for key in info:
        save_object.metadata.add_dictionary({key:info[key]})
    save_object.metadata.add_dictionary({"package":"%s"%info["package"]})
    save_object.metadata.add_dictionary({"reconstruction_type":"%s"%info["ptycho_type"]})
    save_object.metadata.add_dictionary({"R_pixel_size":"%f"%ptycho_recon.sampling[0]})
    save_object.metadata.add_dictionary({"Rx_extent":"%f"%(dataset.R_Ny*dataset.R_pixel_size)})
    save_object.metadata.add_dictionary({"Ry_extent":"%f"%(dataset.R_Nx*dataset.R_pixel_size)})
        
    save_object.save(info["save_path"]+info["data_name"]+"_%s_phase.hspy"%info["ptycho_type"], overwrite=True)
    save_probe.save(info["save_path"]+info["data_name"]+"_%s_probe.hspy"%info["ptycho_type"], overwrite=True)
    
    print("save completed")    


### Use the latest version of 'abTEM' ###
elif info["package"] == "abtem_latest":
    if eval(info["binsize"]) > 1:
        being_processed = np.reshape(being_processed, (bp_shape[0], bp_shape[1], int(bp_shape[2]/eval(info["binsize"])), 
                                                       eval(info["binsize"]), int(bp_shape[3]/eval(info["binsize"])), 
                                                       eval(info["binsize"])))
        being_processed = np.mean(being_processed, axis=(3, 5))
    
    bp_shape = being_processed.shape
    print(bp_shape)
    
    HT = eval(info["HT"])
    scan_step = eval(info["scan_step"])
    center_x, center_y = eval(info["center"])
    wavelength = energy2wavelength(HT)
    print(wavelength)
    reciprocal_step = eval(info["reciprocal_pixel_size"])/wavelength
    print(reciprocal_step)
    
    offset = (-center_x * reciprocal_step, -center_y * reciprocal_step)
    print(offset)
    
    meta = {'energy': HT,
            'semiangle_cutoff': eval(info["semiangle"])*1E3,
            'reciprocal_space': False,
            'label': 'intensity',
            'units': 'arb. unit'}
    
    scan_x = abtem.axes.ScanAxis(sampling=scan_step, label="x", units="Å")
    scan_y = abtem.axes.ScanAxis(sampling=scan_step, label="y", units="Å")
    reci_x = abtem.axes.ReciprocalSpaceAxis(sampling=reciprocal_step)
    reci_y = abtem.axes.ReciprocalSpaceAxis(sampling=reciprocal_step)
    
    axes_meta = [scan_y, scan_x, reci_y, reci_x]
    
    experimental_measurement = DiffractionPatterns.from_array_and_metadata(array=being_processed,
                                                                           axes_metadata=axes_meta,
                                                                           metadata=meta)
    
    print(experimental_measurement)
    print(experimental_measurement.axes_metadata)
    print(experimental_measurement.offset)
    print(experimental_measurement.angular_limits)

    del being_processed
    del original_stack
    
    print(experimental_measurement.shape)
    print(*experimental_measurement.limits, sep="\n")
    for i in range(len(experimental_measurement.axes_metadata)):
        print(experimental_measurement.axes_metadata[i].label, experimental_measurement.axes_metadata[i].units, experimental_measurement.axes_metadata[i].sampling)
    
    # reconstruction sampling scale / extent
    
    experimental_measurement_sampling = tuple(energy2wavelength(HT)*1000/(cal * pixels) 
                                                          for cal,pixels 
                                                          in zip(experimental_measurement.angular_sampling, 
                                                                 experimental_measurement.base_shape))
    
    print(f'expected reconstruction sampling: {experimental_measurement_sampling}')
    
    experimental_measurement_extent = tuple(energy2wavelength(HT)*1000/(cal) 
                                                          for cal
                                                          in experimental_measurement.angular_sampling)
    
    print(f'expected reconstruction extent: {experimental_measurement_extent}')
    
    
    # abTEM latest ptycho - singlislice
    if info["ptycho_type"] == "singleslice":
        print("singleslice ptychography reconstruction process begins")
        ptycho_operator = RegularizedPtychographicOperator(experimental_measurement,
                                                                   semiangle_cutoff=eval(info["semiangle"])*1E3,
                                                                   energy=HT,
                                                                   device=info["device"],
                                                                   parameters={'object_px_padding':(0, 0)}).preprocess()
    
        exp_objects, exp_probes, exp_positions, exp_sse  = ptycho_operator.reconstruct(
            max_iterations = eval(info["num_iteration"]),
            random_seed=1,
            verbose=True,
            return_iterations=True,
            parameters={'alpha':eval(info['alpha']),
                        'beta':eval(info['beta'])})
    
        print("reconstruction completed")
        print("reconstructed real space pixel size (x, y)")
        print(exp_objects[0].sampling)
    
        save_object = hs.signals.Signal2D(np.asarray(exp_objects[-1].array))
        save_probe = hs.signals.Signal2D(np.asarray(exp_probes[-1].array).astype(np.complex64))
    
    # abTEM latest ptycho - mixed state
    if info["ptycho_type"] == "mixed-state":
        print("mixed-state ptychography reconstruction process begins")
        ptycho_operator = MixedStatePtychographicOperator(experimental_measurement,
                                                                   num_probes=eval(info["num_probe"]),
                                                                   semiangle_cutoff=eval(info["semiangle"])*1E3,
                                                                   energy=HT,
                                                                   device=info["device"],
                                                                   parameters={'object_px_padding':(0,0 )}).preprocess()
    
        exp_objects, exp_probes, exp_positions, exp_sse  = ptycho_operator.reconstruct(
            max_iterations = eval(info["num_iteration"]),
            random_seed=1,
            verbose=True,
            return_iterations=True,
            parameters={'alpha':eval(info['alpha']),
                        'beta':eval(info['beta'])})
    
        print("reconstruction completed")
        print("reconstructed real space pixel size (x, y)")
        print(exp_objects[0].sampling)
          
        save_object = hs.signals.Signal2D(np.asarray(exp_objects[-1].array))
        save_probe = hs.signals.Signal2D(np.asarray(exp_probes[-1].array).astype(np.complex64))
    
    # abTEM latest ptycho - multislice
    if info["ptycho_type"] == "multislice":
        print("multislice ptychography reconstruction process begins")
        ptycho_operator = MultislicePtychographicOperator(experimental_measurement,
                                                            semiangle_cutoff=eval(info["semiangle"])*1E3,
                                                            energy=HT,
                                                            num_slices = eval(info["num_slice"]),
                                                            slice_thicknesses = eval(info["slice_thickness"]),
                                                            device=eval(info["defocus"]),
                                                            parameters={'object_px_padding':(0,0)}).preprocess()
    
        exp_objects, exp_probes, exp_positions, exp_sse = ptycho_operator.reconstruct(
            max_iterations = eval(info["num_iteration"]),
            verbose=True,
            random_seed=1,
            return_iterations=True,
            parameters={'alpha':eval(info['alpha']),
                        'beta':eval(info['beta'])})
    
        print("reconstruction completed")
        print("reconstructed real space pixel size (x, y)")
        print(exp_objects[0][0].sampling)
          
        save_object = hs.signals.Signal2D(np.asarray(exp_objects[-1].array))
        save_probe = hs.signals.Signal2D(np.asarray(exp_probes[-1].array).astype(np.complex64))
    
    
    for key in info:
        save_object.metadata.add_dictionary({key:info[key]})
    save_object.metadata.add_dictionary({"package":"%s"%info["package"]})
    save_object.metadata.add_dictionary({"reconstruction_type":"%s"%info["ptycho_type"]})
    save_object.metadata.add_dictionary({"R_pixel_size":"%f"%experimental_measurement_sampling[0]})
    save_object.metadata.add_dictionary({"Rx_extent":"%f"%experimental_measurement.extent[1]})
    save_object.metadata.add_dictionary({"Ry_extent":"%f"%experimental_measurement.extent[0]})
        
    save_object.save(info["save_path"]+info["data_name"]+"_%s_phase.hspy"%info["ptycho_type"], overwrite=True)
    save_probe.save(info["save_path"]+info["data_name"]+"_%s_probe.hspy"%info["ptycho_type"], overwrite=True)
    
    print("save completed")


### Use the legacy version of 'abTEM' ###
else:
    if eval(info["binsize"]) > 1:
        being_processed = np.reshape(being_processed, (bp_shape[0], bp_shape[1], int(bp_shape[2]/eval(info["binsize"])), 
                                                       eval(info["binsize"]), int(bp_shape[3]/eval(info["binsize"])), 
                                                       eval(info["binsize"])))
        being_processed = np.mean(being_processed, axis=(3, 5))
    
    bp_shape = being_processed.shape
    print(bp_shape)
    
    HT = eval(info["HT"])
    scan_step = eval(info["scan_step"])
    semiangle = eval(info["semiangle"])*1E3
    center_x, center_y = eval(info["center"])[0], eval(info["center"])[1]
    wavelength = energy2wavelength(HT)
    reciprocal_step = eval(info["reciprocal_pixel_size"])*1E3
    offset_reci = (-center_x*reciprocal_step, -center_y*reciprocal_step)
    
    x_cb_object = Calibration(offset=0, sampling=scan_step, units="Ã", name="x")
    y_cb_object = Calibration(offset=0, sampling=scan_step, units="Ã", name="y")
    dx_cb_object = Calibration(offset=offset_reci[0], sampling=reciprocal_step, units="mrad", name="alpha_x")
    dy_cb_object = Calibration(offset=offset_reci[1], sampling=reciprocal_step, units="mrad", name="alpha_y")
    
    experimental_measurement = Measurement(being_processed, calibrations=[y_cb_object, x_cb_object, dy_cb_object, dx_cb_object])
    
    print(experimental_measurement.shape)
    print(*experimental_measurement.calibration_limits, sep="\n")
    for i in range(experimental_measurement.dimensions):
        print(experimental_measurement.calibrations[i].name, experimental_measurement.calibrations[i].units, experimental_measurement.calibrations[i].sampling)

    del being_processed
    del original_stack
    
    # reconstruction sampling scale / extent
    
    experimental_measurement_sampling = tuple(energy2wavelength(HT)*1000/(cal.sampling * pixels) 
                                                          for cal,pixels 
                                                          in zip(experimental_measurement.calibrations[-2:], 
                                                                 experimental_measurement.shape[-2:]))
    
    print(f'pixelated_measurement sampling: {experimental_measurement_sampling}')
    
    experimental_measurement_extent = tuple(energy2wavelength(HT)*1000/(cal.sampling) 
                                                          for cal
                                                          in experimental_measurement.calibrations[-2:])
    
    print(f'pixelated_measurement extent: {experimental_measurement_extent}')

    
    
    # abTEM legacy ptycho - singlislice
    if info["ptycho_type"] == "singleslice":
        print("singleslice ptychography reconstruction process begins")
        ptycho_operator = RegularizedPtychographicOperator(experimental_measurement,
                                                                   semiangle_cutoff=eval(info["semiangle"])*1E3,
                                                                   energy=HT,
                                                                   device=info["device"],
                                                                   parameters={'object_px_padding':(0, 0)}).preprocess()

        reconstruction_parameters = {}
        reconstruction_parameters['alpha'] = eval(info['alpha'])
        reconstruction_parameters['beta'] = eval(info['beta'])
        if eval(info['probe_position_correction']):
            reconstruction_parameters['pre_position_correction_update_steps'] = ptycho_operator._num_diffraction_patterns * int(eval(info['pre_position_correction_update_steps']))
            reconstruction_parameters['position_step_size'] = eval(info['position_step_size'])
            reconstruction_parameters['step_size_damping_rate'] = eval(info['step_size_damping_rate'])
    
        exp_objects, exp_probes, exp_positions, exp_sse  = ptycho_operator.reconstruct(
            max_iterations = eval(info["num_iteration"]),
            random_seed=1,
            verbose=True,
            return_iterations=True,
            parameters=reconstruction_parameters)
    
        print("reconstruction completed")
    
        save_object = hs.signals.Signal2D(np.asarray(exp_objects[-1].array))
        save_probe = hs.signals.Signal2D(np.asarray(exp_probes[-1].array).astype(np.complex64))
    
    # abTEM legacy ptycho - mixed state
    if info["ptycho_type"] == "mixed-state":
        print("mixed-state ptychography reconstruction process begins")
        ptycho_operator = MixedStatePtychographicOperator(experimental_measurement,
                                                                   num_probes=eval(info["num_probe"]),
                                                                   semiangle_cutoff=eval(info["semiangle"])*1E3,
                                                                   energy=HT,
                                                                   device=info["device"],
                                                                   parameters={'object_px_padding':(0,0 )}).preprocess()

        reconstruction_parameters = {}
        reconstruction_parameters['alpha'] = eval(info['alpha'])
        reconstruction_parameters['beta'] = eval(info['beta'])
        if eval(info['probe_position_correction']):
            reconstruction_parameters['pre_position_correction_update_steps'] = ptycho_operator._num_diffraction_patterns * eval(info['pre_position_correction_update_steps'])
            reconstruction_parameters['position_step_size'] = eval(info['position_step_size'])
            reconstruction_parameters['step_size_damping_rate'] = eval(info['step_size_damping_rate'])
        
    
        exp_objects, exp_probes, exp_positions, exp_sse  = ptycho_operator.reconstruct(
            max_iterations = eval(info["num_iteration"]),
            random_seed=1,
            verbose=True,
            return_iterations=True,
            parameters=reconstruction_parameters)
    
        print("reconstruction completed")
        
        save_object = hs.signals.Signal2D(np.asarray(exp_objects[-1].array))
        save_probe = hs.signals.Signal2D(np.asarray(exp_probes[-1].array).astype(np.complex64))
    
    # abTEM legacy ptycho - multislice
    if info["ptycho_type"] == "multislice":
        print("multislice ptychography reconstruction process begins")
        ptycho_operator = MultislicePtychographicOperator(experimental_measurement,
                                                            semiangle_cutoff=eval(info["semiangle"])*1E3,
                                                            energy=HT,
                                                            num_slices = eval(info["num_slice"]),
                                                            slice_thicknesses = eval(info["slice_thickness"]),
                                                            device=info["device"],
                                                            parameters={'object_px_padding':(0,0)}).preprocess()

        reconstruction_parameters = {}
        reconstruction_parameters['alpha'] = eval(info['alpha'])
        reconstruction_parameters['beta'] = eval(info['beta'])
        if eval(info['probe_position_correction']):
            reconstruction_parameters['pre_position_correction_update_steps'] = ptycho_operator._num_diffraction_patterns * eval(info['pre_position_correction_update_steps'])
            reconstruction_parameters['position_step_size'] = eval(info['position_step_size'])
            reconstruction_parameters['step_size_damping_rate'] = eval(info['step_size_damping_rate'])
        
        
        exp_objects, exp_probes, exp_positions, exp_sse = ptycho_operator.reconstruct(
            max_iterations = eval(info["num_iteration"]),
            verbose=True,
            random_seed=1,
            return_iterations=True,
            parameters=reconstruction_parameters)
    
        print("reconstruction completed")
         
        save_object = hs.signals.Signal2D(np.asarray(exp_objects[-1].array))
        save_probe = hs.signals.Signal2D(np.asarray(exp_probes[-1].array).astype(np.complex64))
    
    for key in info:
        save_object.metadata.add_dictionary({key:info[key]})
    save_object.metadata.add_dictionary({"package":"%s"%info["package"]})
    save_object.metadata.add_dictionary({"reconstruction_type":"%s"%info["ptycho_type"]})
    save_object.metadata.add_dictionary({"R_pixel_size":"%f"%experimental_measurement_sampling[0]})
    save_object.metadata.add_dictionary({"Rx_extent":"%f"%experimental_measurement.calibration_limits[1][1]})
    save_object.metadata.add_dictionary({"Ry_extent":"%f"%experimental_measurement.calibration_limits[0][1]})
        
    save_object.save(info["save_path"]+info["data_name"]+"_%s_phase.hspy"%info["ptycho_type"], overwrite=True)
    save_probe.save(info["save_path"]+info["data_name"]+"_%s_probe.hspy"%info["ptycho_type"], overwrite=True)
    
    print("save completed")