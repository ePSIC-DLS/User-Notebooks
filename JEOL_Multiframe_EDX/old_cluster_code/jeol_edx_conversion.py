

import matplotlib
matplotlib.use('Agg')
import hyperspy.api as hs
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os

########
# This script does the follwoing:
# - Saves the entire data as hdf5
# - Does a rough alignment, saves the shifts plot and sum image
# - Saves a sum EDX spectrum
#########


def load_apb(apb_filename,frames):
    fd = open(apb_filename, "br")
    file_magic = np.fromfile(fd, "uint8")
    offset = 16668
    data_array = np.zeros((frames,128,128))
    for i in range(frames):
        im_test = file_magic[(i*16384)+offset:(i*16384)+offset+16384]
        reshaped = im_test.reshape(128,128)
        data_array[i,:,:] = reshaped
    hs_data = hs.signals.Signal2D(data_array)
    return(hs_data)


def save_edx_stack(pts_path, apb_path, save_path, data_name, elements_list):
    
    elements_list = sorted(elements_list)
    print('sorted elements list: ', elements_list)

    stack_edx = hs.load(pts_path,lazy=True,sum_frames=False)
    print(hs.__file__)
    print(stack_edx)
    print(f'loaded data {pts_path}')
    stack_edx.save(os.path.join(save_path, data_name), extension='hdf5')
    stack_edx.save(os.path.join(save_path, data_name), extension='zspy')
    
    stack_edx.add_elements(elements_list)
    stack_edx_sum = stack_edx.sum()
    
    stack_edx_sum.save(os.path.join(save_path, data_name + '_sum_edx'))
    stack_edx_sum.plot(True)
    plt.savefig(os.path.join(save_path, data_name + '_sum_edx.png'))
        
    
    stack_apb = load_apb(apb_path,stack_edx.data.shape[0])
    stack_apb_uint8 = stack_apb.data.astype('uint8')
        
    shifts = stack_apb.estimate_shift2D()
    
    f = h5py.File(os.path.join(save_path, data_name + '.hdf5'), 'a')
    f.create_dataset('img_frames', data=stack_apb_uint8, dtype='uint8', compression='gzip')
    f.create_dataset('shifts', data=shifts)
    f.close()
    print(f'Added image frames and shifts to file {os.path.join(save_path, data_name)}')    
    
    plt.figure()
    plt.plot(shifts[:,0], 'b', label='shift_x')
    plt.plot(shifts[:,1], 'r', label='shift_y')
    plt.legend()
    plt.savefig(os.path.join(save_path,f'shifts_{data_name}.png'))
    
    stack_apb.align2D(shifts=shifts)
    

    im_sum = stack_apb.sum()
    
    im_sum.save(os.path.join(save_path , 'image_sum'))
    im_sum.save(os.path.join(save_path , 'image_sum.png'))
    
    # Get the k_factors
    k_fact_file = '/dls/science/groups/e02/Mohsen/code/jupyterhub_active/JEOL_EDX_Multiframe/E01_Kfactors.txt'
    with open(k_fact_file) as f:
        lines = f.readlines()
    k_factor_dict = {}
    for i, line in enumerate(lines):
        if i == 0:
            pass
        else:
            k_factor_dict[line.split()[0]] = {}
            k_factor_dict[line.split()[0]][line.split()[1]] = float(line.split()[2])
    
    k_factors = []
    for element in elements_list:
        k_factors.append(list(k_factor_dict[element].values())[0])
    print(k_factors)
    
    compo = []
    for i in range(stack_edx.data.shape[0]):
        s = stack_edx.inav[:,:,i].sum()
        s.add_elements(elements_list)
        s.add_lines()
        s.compute()
        intensities = s.get_lines_intensity()
        print('intensities: ', intensities)
        atomic_percent = s.quantification(intensities, method='CL',
                                          factors=k_factors)
        compo.append([atomic_percent[c].data[0] for c in range(len(atomic_percent))])
        
    plt.figure()
    compo = np.array(compo)
    for i in range(compo.shape[1]):
        plt.plot(compo[:,i], label = elements_list[i])
    plt.legend()
    plt.savefig(os.path.join(save_path,f'composition_change_{data_name}.png'))

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('elements_list', help='list o felements present in data')
    parser.add_argument('pts_path', help='pts file name')
    parser.add_argument('apb_path', help='apb file name')
    parser.add_argument('output_path', help='path to save files')
    
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                       default=False)

    args = parser.parse_args()
    elements_input = args.elements_list
                          
    
    elements_list = []
    for i, ch in enumerate(elements_input):
        if ch.isalpha() and i < (len(elements_input) -1):
            if elements_input[i+1].isalpha():
                elements_list.append(ch+elements_input[i+1])
            elif not elements_input[i-1].isalpha() or i == 0:
                elements_list.append(ch)
                
    print('elements_list:', elements_list)
    pts_path = args.pts_path
    apb_path = args.apb_path
    
    data_name_blocks = pts_path.split('/')[-4:]
    data_name = '_'.join(data_name_blocks)
    data_name = data_name.split('.')[0]
    print(data_name)
    
            
    save_path = os.path.join(args.output_path, data_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_edx_stack(pts_path, apb_path, save_path, data_name, elements_list)

    
