

import matplotlib
matplotlib.use('Agg')
import hyperspy.api as hs
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os
import json
import gc

###
# This script does the following:
# - reads the hdf5 file with EDX and image stacks
# - Performs an alignmnet on the image stack
# - Based on the index ranges provided the maps are aligned and summed
# - Option to output 3D spectrum image
###


def process_edx_stack(data_path, elements_list, lines_list, ind_to_include, output_path, output_3D, shifts_file):
    
    min_batch_size = 100
    with h5py.File(data_path, 'r') as f:
        im_stack = f['img_frames'][()]
        if 'shifts' in f.keys():
            shifts = f['shifts'][()]
    if os.path.exists(shifts_file):
        shifts = np.load(shifts_file)
        
    im_stack = im_stack.astype('uint8')
    im_stack = hs.signals.Signal2D(im_stack)
    
    if im_stack.data.shape[0] < min_batch_size:
        batch_size = im_stack.data.shape[0]
    else:
        batch_size = min_batch_size

    elements_list = sorted(elements_list)
    lines_list = sorted(lines_list)
    
    stack_edx = hs.load(data_path, lazy=True)
    print(f'Lazily loaded data {data_path}')
    if ind_to_include != "":
        stack_edx = stack_edx.inav[:,:,ind_to_include[0]:ind_to_include[1]]
        shifts = shifts[ind_to_include[0]:ind_to_include[1]]
        print('after cropping: ', stack_edx)
    
    binned_eds = stack_edx.rebin(scale=(4,4,1,2))
    binned_eds.add_elements(elements_list)
    binned_eds.add_lines(lines_list)
    
    batch_num = binned_eds.data.shape[0] // batch_size
    print(f'batch_number: {batch_num}')
    
    if output_3D:
        si_sum = np.zeros((2048,128,128))
    
    for i in range(batch_num):
        batch_to_compute = binned_eds.inav[:,:,int(i*batch_size):int(i*batch_size + batch_size)]
        batch_to_compute.compute(parallel=True, max_workers=8)
        
        print(f'Computed the binned version of the EDX stack_batch number {i}')

        maps = batch_to_compute.get_lines_intensity()
    
        stack_list = []
        stack_names = []
        for cnt, element in enumerate(elements_list):
            stack_list.append(maps[cnt].as_signal2D(2))
            stack_names.append(element + '_stack')
    
    
        im_stack_batch = im_stack.inav[int(i*batch_size):int(i*batch_size + batch_size)]
        im_stack_batch.align2D(shifts=shifts[int(i*batch_size):int(i*batch_size + batch_size)])
    
        stack_sum = []
        for stack in stack_list:
            stack.align2D(shifts=shifts[int(i*batch_size):int(i*batch_size + batch_size)])
            stack_sum.append(stack.sum())

        im_sum = im_stack.sum()
    
        for cnt, el_map in enumerate(stack_sum):
            el_map.save(os.path.join(output_path, stack_names[cnt] + f'_ind{i*batch_size}_to_{int(i*batch_size + batch_size)}'))
            el_map.save(os.path.join(output_path, stack_names[cnt] + f'_ind{i*batch_size}_to_{int(i*batch_size + batch_size)}.png'))
    
        im_sum.save(os.path.join(output_path , f'image_sum_ind{i*batch_size}_to_{int(i*batch_size + batch_size)}'))
        im_sum.save(os.path.join(output_path , f'image_sum_ind{i*batch_size}_to_{int(i*batch_size + batch_size)}.png'))
        
        if output_3D:
            # si_sum = np.zeros((2048,128,128))
            shifts_sub = shifts[int(i*batch_size):int(i*batch_size + batch_size)]
            for d in range(batch_to_compute.data.shape[0]):
                si = batch_to_compute.inav[:,:,d]
                si_shifted = shift_si(si, -shifts_sub[d])
                si_sum = si_sum + si_shifted.data
            
        del batch_to_compute
        gc.collect()
        
    if output_3D:
        si_sum = hs.signals.Signal2D(si_sum)
        si_sum = si_sum.T
        edx_hs = hs.signals.EDSTEMSpectrum(si_sum)
        
        edx_hs.axes_manager[0].name = binned_eds.axes_manager[0].name
        edx_hs.axes_manager[1].name = binned_eds.axes_manager[1].name
        edx_hs.axes_manager[2].name = binned_eds.axes_manager[3].name
        
        edx_hs.axes_manager[0].scale = binned_eds.axes_manager[0].scale
        edx_hs.axes_manager[1].scale = binned_eds.axes_manager[1].scale
        edx_hs.axes_manager[0].offset = binned_eds.axes_manager[0].offset
        edx_hs.axes_manager[1].offset = binned_eds.axes_manager[1].offset
        edx_hs.axes_manager[0].units = binned_eds.axes_manager[0].units
        edx_hs.axes_manager[1].units = binned_eds.axes_manager[1].units

        edx_hs.axes_manager[2].scale = binned_eds.axes_manager[3].scale
        edx_hs.axes_manager[2].offset = binned_eds.axes_manager[3].offset
        edx_hs.axes_manager[2].units = binned_eds.axes_manager[3].units
        
        edx_hs.add_elements(elements_list)
        edx_hs.add_lines(lines_list)
        
        edx_hs.save(os.path.join(output_path , f'SI_sum'))
    
from scipy import ndimage
def shift_image(im, shift=0, interpolation_order=1, fill_value=np.nan):
    if not np.any(shift):
        return im
    else:
        fractional, integral = np.modf(shift)
        if fractional.any():
            order = interpolation_order
        else:
            # Disable interpolation
            order = 0
        return ndimage.shift(im, shift, cval=fill_value, order=order)
    

def shift_si(si, shift):
    """
    si is a hyperspy EDX object
    """
    from functools import partial
    mapfunc = partial(shift_image, shift=shift)
    si_t = si.T
    si_shift = map(mapfunc, si_t.data)
    si_shift = list(si_shift)
    si_shift = np.asarray(si_shift)
    si_shift = si_shift.astype('uint8')
    return hs.signals.Signal2D(si_shift)
    

if __name__ =='__main__':
    from multiprocessing.pool import ThreadPool
    import dask
    dask.config.set(pool=ThreadPool(8))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help='json file path')

    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                       default=False)

    args = parser.parse_args()
    with open(args.json_path) as f:
        json_data = json.load(f)
    
    data_path = json_data['data_path']
    elements_list = json_data['element_list']
    lines_list = json_data['x-ray_line_list']
    ind_to_include = json_data['ind_to_include']
    output_3D = json_data['3D_output']
    output_path = json_data['output_path']
    shifts_file = json_data['shifts_file']
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    process_edx_stack(data_path, elements_list, lines_list, ind_to_include, output_path, output_3D, shifts_file)
    
