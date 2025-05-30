# Python script to submit a slurm job for ptychography
# PtyREX / abTEM (latest or legacy) / py4DSTEM
# singleslice / mixed-state / multislice (unavailable for abTEM latest) / mixed-state multislice
# Created by Jinseok Ryu (jinseok.ryu@diamond.ac.uk)

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import json
import py4DSTEM
import hyperspy.api as hs

print(py4DSTEM.__version__)


class ptycho_submit:
    def __init__(self, data_path, meta_path=None, mask_path=None, save_dir=None, device='cpu'):
        self.data_path = data_path
        self.data_name = data_path.split("/")[-1].split(".")[0]
        self.device = device
        self.mask_path = mask_path
        self.meta_path = meta_path
        self.save_dir = save_dir
        
        if meta_path == None:
            try:
                self.meta_path = data_path[:-10]+".hdf"
            except:
                self.meta_path = None
        if mask_path == None:
            self.mask_path = '/dls/science/groups/e02/Ryu/RYU_at_ePSIC/python_ptycho/mask/29042024_12bitmask.h5'
        if save_dir == None:
            self.save_dir = os.path.dirname(data_path)
        
        print("data_path:"+self.data_path)
        print("data_name:"+self.data_name)
        print("meta_path:"+self.meta_path)
        print("mask_path:"+self.mask_path)
        
        if self.meta_path != None:
            with h5.File(self.meta_path,'r') as f:
                print("----------------------------------------------------------")
                print(f['metadata']["defocus(nm)"])
                print(f['metadata']["defocus(nm)"][()])
                self.defocus_exp = f['metadata']["defocus(nm)"][()]*10 # Angstrom
                print("----------------------------------------------------------")
                print(f['metadata']["ht_value(V)"])
                print(f['metadata']["ht_value(V)"][()])
                self.HT = f['metadata']["ht_value(V)"][()]
                print("----------------------------------------------------------")
                print(f['metadata']["step_size(m)"])
                print(f['metadata']["step_size(m)"][()])
                self.scan_step = f['metadata']["step_size(m)"][()] * 1E10 # Angstrom
                print("----------------------------------------------------------")
                print(f['metadata']['nominal_camera_length(m)'])
                print(f['metadata']['nominal_camera_length(m)'][()])
                self.camera_length = f['metadata']['nominal_camera_length(m)'][()]
                print("----------------------------------------------------------")
                print(f['metadata']['aperture_size'])
                print(f['metadata']['aperture_size'][()])
                self.aperture = f['metadata']['aperture_size'][()]
                print("----------------------------------------------------------")

                self.rotation_angle_exp, self.semiangle = Meta2Config(self.HT, self.camera_length, self.aperture)
                
        else:
            self.defocus_exp = 0
            self.HT = None
            self.scan_step = None
            self.camera_length = None
            self.aperture = None
            self.rotation_angle_exp = 0
            self.semiangle = None

        print("defocus_exp:", self.defocus_exp)  
        print("HT:", self.HT)
        print("scan_step:", self.scan_step) 
        print("camera_length:", self.camera_length) 
        print("aperture:", self.aperture) 
        print("rotation_angle_expT:", self.rotation_angle_exp) 
        print("semiangle:", self.semiangle) 
        
        
        if self.mask_path != None:
            try:
                with h5.File(self.mask_path,'r') as f:
                    mask = f['data']['mask'][()]
                mask = np.bool_(mask)
                mask = np.invert(mask)
                mask = mask.astype(np.float32)

            except:
                with h5.File(self.mask_path,'r') as f:
                    mask = f['root']['np.array']['data'][()]
                mask = np.bool_(mask)
                mask = np.invert(mask)
                mask = mask.astype(np.float32)
                
            self.mask = mask

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(self.mask)
            fig.tight_layout()
            plt.show()
        
        else:
            self.mask = None
            
            
    def data_load(self, semiangle=None, defocus_exp=None):
        if semiangle != None:
            self.semiangle = semiangle
        if defocus_exp != None:
            self.defocus_exp = defocus_exp
            
        if self.data_path.split(".")[-1] == "hspy": 
        # This is for the simulated 4DSTEM data using 'submit_abTEM_4DSTEM_simulation.ipynb'
        # stored in /dls/science/groups/e02/Ryu/RYU_at_ePSIC/multislice_simulation/submit_abtem/submit_abtem_4DSTEM_simulation.ipynb
            self.original_stack = hs.load(self.data_path)
            print(self.original_stack)
            n_dim = len(self.original_stack.data.shape)
            scale = []
            origin = []
            unit = []
            size = []

            for i in range(n_dim):
                print(self.original_stack.axes_manager[i].scale, self.original_stack.axes_manager[i].offset, self.original_stack.axes_manager[i].units, self.original_stack.axes_manager[i].size)
                scale.append(self.original_stack.axes_manager[i].scale)
                origin.append(self.original_stack.axes_manager[i].offset)
                unit.append(self.original_stack.axes_manager[i].units)
                size.append(self.original_stack.axes_manager[i].size)

            self.HT = eval(self.original_stack.metadata["HT"])
            if self.HT < 1000:
                self.HT *= 1000
            try:
                self.defocus_exp = eval(self.original_stack.metadata["defocus"])
                self.semiangle = eval(self.original_stack.metadata["semiangle"])
            except:
                print("No metadata found")
            self.scan_step = scale[0]
            print("HT: ", self.HT)
            print("experimental defocus: ", self.defocus_exp)
            print("semiangle: ", self.semiangle)
            print("scan step: ", self.scan_step)
            self.original_stack = self.original_stack.data
            self.det_name = 'ePSIC_EDX'
            self.data_key = 'Experiments/__unnamed__/data'


        elif self.data_path.split(".")[-1] == "hdf" or self.data_path.split(".")[-1] == "hdf5" or self.data_path.split(".")[-1] == "h5":
            # try:
            #     original_stack = hs.load(data_path, reader="HSPY", lazy=True)
            #     print(original_stack)
            #     original_stack = original_stack.data
            try:    
                f = h5.File(self.data_path,'r')
                print(f)
                self.original_stack = f['Experiments']['__unnamed__']['data'][:]
                f.close()
                self.det_name = 'ePSIC_EDX'
                self.data_key = 'Experiments/__unnamed__/data'

            except:
                f = h5.File(self.data_path,'r')
                print(f)
                self.original_stack = f['data']['frames'][:]
                f.close()
                self.det_name = 'pty_data'
                self.data_key = "data/frames"

        elif self.data_path.split(".")[-1] == "dm4":
            self.original_stack = hs.load(self.data_path)
            print(self.original_stack)
            n_dim = len(self.original_stack.data.shape)
            scale = []
            origin = []
            unit = []
            size = []

            for i in range(n_dim):
                print(self.original_stack.axes_manager[i].scale, self.original_stack.axes_manager[i].offset, self.original_stack.axes_manager[i].units, self.original_stack.axes_manager[i].size)
                scale.append(self.original_stack.axes_manager[i].scale)
                origin.append(self.original_stack.axes_manager[i].offset)
                unit.append(self.original_stack.axes_manager[i].units)
                size.append(self.original_stack.axes_manager[i].size)

            self.HT = 1000 * self.original_stack.metadata['Acquisition_instrument']['TEM']['beam_energy']
            self.scan_step = scale[0] * 10
            print("HT: ", self.HT)
            print("experimental defocus: ", self.defocus_exp)
            print("semiangle: ", self.semiangle)
            print("scan step: ", self.scan_step)
            self.original_stack = self.original_stack.data
            self.original_stack = self.original_stack.astype(np.float32)
            self.original_stack -= np.min(self.original_stack)
            self.original_stack /= np.max(self.original_stack)
            self.original_stack *= 128.0
            # det_name = 'ePSIC_EDX'
            # data_key = 'Experiments/__unnamed__/data'    

        else:
            print("Wrong data format!")

        self.original_stack = self.original_stack.astype(np.float32)
        print(self.original_stack.dtype)
        print(self.original_stack.shape)
        print(np.min(self.original_stack), np.max(self.original_stack))
        
        if self.mask_path != None and type(self.mask) == np.ndarray:
            for i in range(self.original_stack.shape[0]):
                for j in range(self.original_stack.shape[1]):
                    self.original_stack[i, j] = np.multiply(self.original_stack[i, j], self.mask)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.original_stack.sum(axis=(2,3)), cmap='inferno')
        ax[1].imshow(self.original_stack[0, 0], cmap='jet')
        fig.tight_layout()
        plt.show()
        
        
    def scan_region_crop(self, sy, sx, width, crop_R=False):
        self.crop_R_region = (sy,sy+width,sx,sx+width)
        print(self.original_stack[self.crop_R_region[0]:self.crop_R_region[1], self.crop_R_region[2]:self.crop_R_region[3]].shape)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(self.original_stack[self.crop_R_region[0]:self.crop_R_region[1], self.crop_R_region[2]:self.crop_R_region[3]].sum(axis=(2,3)), cmap='inferno')
        fig.tight_layout()
        plt.show()
        
        self.crop_R = crop_R
        
        if crop_R:
            self.being_processed = self.original_stack[self.crop_R_region[0]:self.crop_R_region[1], self.crop_R_region[2]:self.crop_R_region[3]]
            self.bp_shape = self.being_processed.shape

        else:
            self.being_processed = self.original_stack
            self.bp_shape = self.being_processed.shape

        print(self.bp_shape)
        
    def DP_region_crop(self, sy, sx, width, crop_Q=False):
        self.crop_Q_region = (sy,sy+width,sx,sx+width)
        print(self.original_stack[0, 0, self.crop_Q_region[0]:self.crop_Q_region[1], self.crop_Q_region[2]:self.crop_Q_region[3]].shape)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(self.original_stack[0, 0, self.crop_Q_region[0]:self.crop_Q_region[1], self.crop_Q_region[2]:self.crop_Q_region[3]], cmap='jet')
        fig.tight_layout()
        plt.show()

        self.crop_Q = crop_Q
        
        if crop_Q:
            self.being_processed = self.being_processed[:, :, self.crop_Q_region[0]:self.crop_Q_region[1], self.crop_Q_region[2]:self.crop_Q_region[3]]
            self.bp_shape = self.being_processed.shape

        else:
            self.bp_shape = self.being_processed.shape

        print(self.bp_shape)
        
    def binning(self, binsize=1):
        self.binsize = binsize
        if self.bp_shape[3] % self.binsize != 0:
            remove_ = int(self.bp_shape[3] % self.binsize)
            self.being_processed = self.being_processed[:, :, :-remove_, :-remove_]
            self.bp_shape = self.being_processed.shape
        print(self.bp_shape)
        
    def fill_cross(self, fill_region=None, num_neighbor=7, fill=False):
        self.fill = fill
        self.fill_region = fill_region
        self.num_neighbor = num_neighbor
        if self.fill:
            zero_ind = np.where(self.mask[self.fill_region[0]:self.fill_region[1], self.fill_region[2]:self.fill_region[3]]==0)
            test = force_fill(self.original_stack[0, 0, self.fill_region[0]:self.fill_region[1], self.fill_region[2]:self.fill_region[3]], zero_ind[0], zero_ind[1], num_neighbor=num_neighbor)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(self.mask[self.fill_region[0]:self.fill_region[1], self.fill_region[2]:self.fill_region[3]])
            ax[1].imshow(self.original_stack[0, 0, self.fill_region[0]:self.fill_region[1], self.fill_region[2]:self.fill_region[3]])
            ax[2].imshow(test)
            fig.tight_layout()
            plt.show()
             
            for i in range(self.bp_shape[0]):
                for j in range(self.bp_shape[1]):
                        self.being_processed[i, j][self.fill_region[0]:self.fill_region[1], self.fill_region[2]:self.fill_region[3]] = force_fill(self.being_processed[i, j, self.fill_region[0]:self.fill_region[1], self.fill_region[2]:self.fill_region[3]], zero_ind[0], zero_ind[1], num_neighbor=7)


                        
    def disk_detect(self, delete_stack=True, th_lower=0.01, th_upper=0.20, reload=False):
        try:
            if reload:
                self.dataset = py4DSTEM.DataCube(data=self.being_processed)
                print("original dataset")
                print(self.dataset)
                if delete_stack:
                    del self.being_processed # to reduce the memory usage
                    del self.original_stack # to reduce the memory usage

                if self.binsize > 1:
                    self.dataset.bin_Q(self.binsize)
                    print("after binning")
                    print(self.dataset)

                self.dataset.get_dp_mean()
                self.dataset.get_dp_max()      
                
            py4DSTEM.show([self.dataset.tree('dp_mean')],
                    cmap='jet',
                    figsize=(4, 4))
        
            self.probe_radius_pixels, self.probe_qx0, self.probe_qy0 = self.dataset.get_probe_size(thresh_lower=th_lower, thresh_upper=th_upper, N=100, plot=True)

            self.dataset.calibration._params['Q_pixel_size'] = self.semiangle / self.probe_radius_pixels
            self.dataset.calibration._params['Q_pixel_units'] = "mrad"
            self.dataset.calibration._params['R_pixel_size'] = self.scan_step
            self.dataset.calibration._params['R_pixel_units'] = "A"

            print(self.dataset)
            print(self.dataset.calibration)
            
        except:
            self.dataset = py4DSTEM.DataCube(data=self.being_processed)
            print("original dataset")
            print(self.dataset)
            if delete_stack:
                del self.being_processed # to reduce the memory usage
                del self.original_stack # to reduce the memory usage

            if self.binsize > 1:
                self.dataset.bin_Q(self.binsize)
                print("after binning")
                print(self.dataset)

            self.dataset.get_dp_mean()
            self.dataset.get_dp_max()

            py4DSTEM.show([self.dataset.tree('dp_mean')],
                    cmap='jet',
                    figsize=(4, 4))
        
            self.probe_radius_pixels, self.probe_qx0, self.probe_qy0 = self.dataset.get_probe_size(thresh_lower=th_lower, thresh_upper=th_upper, N=100, plot=True)
            
            self.dataset.calibration._params['Q_pixel_size'] = self.semiangle / self.probe_radius_pixels
            self.dataset.calibration._params['Q_pixel_units'] = "mrad"
            self.dataset.calibration._params['R_pixel_size'] = self.scan_step
            self.dataset.calibration._params['R_pixel_units'] = "A"

            print(self.dataset)
            print(self.dataset.calibration)
            
        light_speed = 299792458 # speed of light [m/s]
        m0 = 9.1093837E-31 # mass of an electron [kg]
        planck = 6.62607015E-34 # h [m^2*kg/s]
        e_volt = 1.602176634E-19 # eV [m^2*kg/s^2]
        self.wavelength = planck/np.sqrt(2*m0*self.HT*e_volt*(1+self.HT*e_volt/(2*m0*light_speed**2)))*1E10

        self.R_extent = self.dataset.calibration.R_pixel_size * self.dataset.shape[0]
        self.k_extent = self.dataset.calibration.Q_pixel_size * self.dataset.shape[2]
        self.recon_R_pixel_size = self.wavelength / self.k_extent * 1000
        self.recon_R_extent = self.wavelength / self.dataset.calibration.Q_pixel_size * 1000

        print("scan step size: %f Å"%self.dataset.calibration.R_pixel_size)
        print("reconstructed pixel size: %f Å"%self.recon_R_pixel_size)
        print("scan extent: %f Å"%self.R_extent)
        print("reconstructed extent: %f Å"%self.recon_R_extent)
        
        plt.show()
            
            
    def virtual_STEM(self):
        center = (self.probe_qx0, self.probe_qy0)
        self.radius_BF = self.probe_radius_pixels
        self.radii_DF = (self.probe_radius_pixels*1.1, int(self.dataset.Q_Nx/2))

        self.dataset.get_virtual_image(
            mode = 'circle',
            geometry = (center,self.radius_BF),
            name = 'bright_field',
            shift_center = False,
        )
        self.dataset.get_virtual_image(
            mode = 'annulus',
            geometry = (center,self.radii_DF),
            name = 'dark_field',
            shift_center = False,
        )

        py4DSTEM.show([self.dataset.tree('bright_field'),
                        self.dataset.tree('dark_field')],
                    cmap='inferno',
                    figsize=(10, 10))
        plt.show()
        
        
    def DPC(self, low_pass=None, high_pass=None):
        self.dpc = py4DSTEM.process.phase.DPC(
        datacube=self.dataset,
        energy=self.HT).preprocess(force_com_rotation=self.rotation_angle_exp,
                                 force_com_transpose=False)
        plt.show()
    
        self.dpc.reconstruct(
            max_iter=8,
            store_iterations=True,
            reset=True,
            gaussian_filter_sigma=0.2,
            gaussian_filter=True,
            q_lowpass=low_pass,
            q_highpass=high_pass
        ).visualize(
            iterations_grid='auto',
            figsize=(16, 10)
        )
        plt.show()
        
        self.dpc_cor = py4DSTEM.process.phase.DPC(
            datacube=self.dataset,
            energy=self.HT,
            verbose=False,
        ).preprocess(
            force_com_rotation=np.rad2deg(self.dpc._rotation_best_rad),
            force_com_transpose=False,
        )
        plt.show()
        
        self.dpc_cor.reconstruct(
            max_iter=8,
            store_iterations=True,
            reset=True,
            gaussian_filter_sigma=0.2,
            gaussian_filter=True,
            q_lowpass=low_pass,
            q_highpass=high_pass
        ).visualize(
            iterations_grid='auto',
            figsize=(16, 10)
        )
        plt.show()
        
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(self.dpc._com_normalized_y, cmap="bwr")
        ax[0].set_title("CoMx")
        ax[1].imshow(self.dpc._com_normalized_x, cmap="bwr")
        ax[1].set_title("CoMy")
        ax[2].imshow(np.sqrt(self.dpc._com_normalized_y**2 + self.dpc._com_normalized_x**2), cmap="inferno")
        ax[2].set_title("Magnitude of CoM")
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(self.dpc_cor._com_normalized_y, cmap="bwr")
        ax[0].set_title("CoMx - rotation corrected")
        ax[1].imshow(self.dpc_cor._com_normalized_x, cmap="bwr")
        ax[1].set_title("CoMy - rotation corrected")
        ax[2].imshow(np.sqrt(self.dpc_cor._com_normalized_y**2 + self.dpc_cor._com_normalized_x**2), cmap="inferno")
        ax[2].set_title("Magnitude of CoM - rotation corrected")
        fig.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(self.dpc.object_phase, cmap="inferno")
        ax[0].set_title("iCoM")
        ax[1].imshow(self.dpc_cor.object_phase, cmap="inferno")
        ax[1].set_title("iCoM - rotation corrected")
        fig.tight_layout()
        plt.show()
        

    def parallax(self):
        
        self.parallax = py4DSTEM.process.phase.Parallax(
                    datacube=self.dataset,
                    energy = self.HT,
                    device = self.device, 
                    verbose = True
                ).preprocess(
                    normalize_images=True,
                    plot_average_bf=False,
                    edge_blend=8,
                )
        
        self.parallax = self.parallax.reconstruct(
                    reset=True,
                    regularizer_matrix_size=(1,1),
                    regularize_shifts=True,
                    running_average=True,
                    min_alignment_bin = 2,
                    num_iter_at_min_bin = 4,
                )
        plt.show()
        
        self.parallax.show_shifts()
        plt.show()

        self.parallax.subpixel_alignment(
            #kde_upsample_factor=2,
            kde_sigma_px=0.125,
            plot_upsampled_BF_comparison=True,
            plot_upsampled_FFT_comparison=True,
        )
        plt.show()
        
        self.parallax.aberration_fit(
            plot_CTF_comparison=True,
        )
        plt.show()

        self.parallax.aberration_correct(figsize=(5, 5))
        plt.show()
        
        self.semiangle_cutoff_estimated = self.dataset.calibration.get_Q_pixel_size() * self.probe_radius_pixels
        print('semiangle cutoff estimate = ' + str(np.round(self.semiangle_cutoff_estimated, decimals=1)) + ' mrads')

        self.defocus_estimated = -parallax.aberration_C1
        print('estimated defocus         = ' + str(np.round(self.defocus_estimated)) + ' Angstroms')

        self.rotation_degrees_estimated = np.rad2deg(parallax.rotation_Q_to_R_rads)
        print('estimated rotation        = ' + str(np.round(self.rotation_degrees_estimated)) + ' deg')


        
    def prepare_submit(self, script_path=None, package='py4dstem', 
                       ptycho_type='mixed-state-multislice', num_iteration=30, 
                       num_probe=6, num_slice=10, slice_thickness=20,
                      alpha=0.5, beta=0.5, reconstruction_parameter=1.0,
                       shift_radius=0.5, shift_trial=3,
                       max_batch_size=256, tv_denoise=False):
        
        semiangle_cutoff = self.semiangle
        defocus = self.defocus_exp
        rotation_degrees = self.rotation_angle_exp
        
        self.package = package
        
        # generate the information file
        # Please note that
        # pyfftw (0.13.1) causes annoying warnings in the reconstruction process using abtem.
        # pyfftw (0.12.0) does not cause any warnings -> I use my own python environment and you can access too.
        # The reconstructed object and probe will be saved as .hspy files - able to see via DAWN
        # package = "ptyrex", "abtem_latest", "abtem_legacy", or "py4dstem"
        
        if package == 'ptyrex':
            device = "gpu" # "cpu" or "gpu"
            gpu_type = "pascal" # "pascal" or "volta"
            gpu_node = 4
        else:
            device = "gpu" # "cpu" or "gpu"
            gpu_type = "volta" # "pascal" or "volta"
            gpu_node = 1

        data_name = self.data_path.split("/")[-1].split(".")[0]
        data_name = time.strftime("%Y%m%d_%H%M%S") + "_" + data_name
        if script_path == None:
            script_path = '/dls/science/groups/e02/Ryu/RYU_at_ePSIC/python_ptycho/python_ptycho.py'

        save_path = self.save_dir + '/%s_ptycho/'%package
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Save directory created: "+save_path)
        if package == "ptyrex":
            cal_name = data_name+"_calibration_info.json"
        else:
            cal_name = data_name+"_calibration_info.txt"
        sub_name = data_name+"_submit.sh"
        log_path = save_path + data_name + "_"
        
        self.save_path = save_path
        self.sub_name = sub_name
        self.cal_name = cal_name

        # ptycho_type = "multislice" # "singleslice", "mixed-state", "multislice" or "mixed-state-multislice"
        # num_iteration = 30
        # num_probe = 4 # for mixed-state ptychography or mixed-state multislice ptychography
        # num_slice = 12 # for multislice ptychography or mixed-state multislice ptychography
        # slice_thickness = 20 # for multislice ptychography or mixed-state multislice ptychography

        # PtyRex reconstruction parameters
        camera_length = (1000 * 0.000055 / (self.semiangle/(self.binsize*self.probe_radius_pixels)))
        ptyrex_crop = list(self.dataset.shape[2:]) # []
        # shift_radius = 0.5
        # shift_trial = 3

        if ptycho_type == "multislice" or ptycho_type == "mixed-state-multislice":
            ptyrex_template = '/dls/science/groups/e02/Ryu/RYU_at_ePSIC/python_ptycho/ptyrex_template/multislice_example.json'
        elif ptycho_type == "singleslice":
            ptyrex_template = '/dls/science/groups/e02/Ryu/RYU_at_ePSIC/python_ptycho/ptyrex_template/singleslice_example.json'
        else:
            print("Wrong ptychography type!")

        #data_key = "" # 'Experiments/__unnamed__/data', "data/frames" or ''

        # abTEM recontruction parameters
        # alpha = 0.5 # also used in PtyREX
        # beta = 0.5 # also used in PtyREX
        probe_position_correction = False
        pre_position_correction_update_steps = 1
        position_step_size = 0.5
        step_size_damping_rate = 0.995

        # py4DSTEM reconstruction parameters
        # max_batch_size = 256
        reconstruction_method = "GD"
        # reconstruction_parameter = 1.0
        normalization_min = 1.0
        identical_slices = False
        object_positivity = False
        # tv_denoise = False
        tv_denoise_weights = [0.1,0.1]
        tv_denoise_inner_iter = 40
        tv_denoise_chambolle = False
        tv_denoise_weight_chambolle = 0.01

        # Expected memory necessary for reconstruction
        print("data shape = ", self.dataset.shape)
        num_element = np.prod(self.dataset.shape)
        dp_element = np.prod(self.dataset.shape[2:])

        #print("memory for the data array (float32) = %.3f Bytes"%(num_element*4))
        #print("memory for the data array (float32) = %.3f Gb"%(num_element*4/2**30))

        #print("memory for the data array (float64) = %.3f Bytes"%(num_element*8))
        #print("memory for the data array (float64) = %.3f Gb"%(num_element*8/2**30))

        print("memory for the data array (complex64) = %.3f Bytes"%(num_element*8))
        print("memory for the data array (complex64) = %.3f Gb"%(num_element*8/2**30))

        print("memory for the data array (complex128) = %.3f Bytes"%(num_element*16))
        print("memory for the data array (complex128) = %.3f Gb"%(num_element*16/2**30))

        print("memory for the data array (complex64) - number of slices considered = %.3f Bytes"%(num_element*8*num_slice))
        print("memory for the data array (complex64) - number of slices considered = %.3f Gb"%(num_element*8*num_slice/2**30))

        print("memory for the data array (complex128) - number of slices considered = %.3f Bytes"%(num_element*16*num_slice))
        print("memory for the data array (complex128) - number of slices considered = %.3f Gb"%(num_element*16*num_slice/2**30))

        print("***************************************")
        print("abTEM needs more than %.3f Gb memory"%(num_element*8/2**30))
        print("py4DSTEM needs more than %.3f Gb memory without max batch"%(num_element*8*num_slice/2**30))
        print("py4DSTEM needs more than %.3f Gb memory with max batch"%(max_batch_size*dp_element*8*num_slice/2**30))
        
        
        if package == "ptyrex":
            with open(ptyrex_template,'r') as f:
                pty_expt = json.load(f)

            pty_expt['base_dir'] = save_path
            pty_expt['process']['save_dir'] = save_path
            pty_expt['process']['common']['scan']['rotation'] = rotation_degrees

            pty_expt['experiment']['experiment_ID'] = data_name
            pty_expt['experiment']['data']['data_path'] = self.data_path
            pty_expt['experiment']['data']['data_key'] = self.data_key
            if self.mask_path != None:
                pty_expt['experiment']['data']['dead_pixel_flag'] = 1
                pty_expt['experiment']['data']['dead_pixel_path'] = self.mask_path
            else:
                pty_expt['experiment']['data']['dead_pixel_flag'] = 0
            pty_expt['experiment']['detector']['position'] = [0, 0, camera_length]
            pty_expt['experiment']['optics']['lens']['alpha'] = 2*semiangle_cutoff*1E-3
            pty_expt['experiment']['optics']['lens']['defocus'] = [defocus*1E-10, defocus*1E-10]

            pty_expt['process']['common']['scan']['N'] = [self.bp_shape[0], self.bp_shape[1]]
            pty_expt['process']['common']['scan']['dR'] = [self.scan_step*1E-10, self.scan_step*1E-10]
            if self.mask_path != None:
                pty_expt['process']['common']['detector']['mask_flag'] = 1
            else:
                pty_expt['process']['common']['detector']['mask_flag'] = 0
            pty_expt['process']['common']['detector']['bin'] = [self.binsize, self.binsize]
            if ptyrex_crop != []:
                pty_expt['process']['common']['detector']['crop'] = ptyrex_crop
            else:
                pty_expt['process']['common']['detector']['crop'] = [self.bp_shape[2], self.bp_shape[3]]
            pty_expt['process']['common']['detector']['name'] = self.det_name
            pty_expt['process']['common']['probe']['convergence'] = 2*semiangle_cutoff*1E-3
            pty_expt['process']['common']['source']['energy'] = [self.HT]
            pty_expt['process']['save_prefix'] = data_name
            pty_expt['process']['PIE']['iterations'] = num_iteration
            pty_expt['process']['PIE']['object']['alpha'] = alpha
            pty_expt['process']['PIE']['probe']['alpha'] = beta
            pty_expt['process']['PIE']['scan']['shift radius'] = shift_radius
            pty_expt['process']['PIE']['scan']['shift trials'] = shift_trial
            if ptycho_type == "mixed-state-multislice" or ptycho_type == "mixed-state":
                pty_expt['process']['PIE']['source']['sx'] = num_probe
            else:
                pty_expt['process']['PIE']['source']['sx'] = 1

            if ptycho_type == "multislice" or ptycho_type == "mixed-state-multislice":
                pty_expt['process']['PIE']['MultiSlice']['S_distance'] = slice_thickness * 1E-10
                pty_expt['process']['PIE']['MultiSlice']['slices'] = num_slice


            with open(save_path+cal_name, 'w') as f:
                json.dump(pty_expt, f, indent=4)

        else:
            with open(save_path+cal_name, 'w') as f:
                f.write("package : "+package+"\n")
                f.write("data_path : "+self.data_path+"\n")
                f.write("data_name : "+data_name+"\n")
                f.write("mask_path : "+self.mask_path+"\n")
                f.write("save_path : "+save_path+"\n")
                f.write("device : "+device+"\n")
                f.write("HT : %f\n"%self.HT)
                f.write("scan_step : %f"%self.scan_step+"\n")
                if self.crop_R:
                    f.write("crop_R : "+"True"+"\n")
                    f.write("crop_R_region : "+"(%d,%d,%d,%d)"%(self.crop_R_region[0], self.crop_R_region[1], self.crop_R_region[2], self.crop_R_region[3])+"\n")
                else:
                    f.write("crop_R : "+"False"+"\n")
                if self.crop_Q:
                    f.write("crop_Q : "+"True"+"\n")
                    f.write("crop_Q_region : "+"(%d,%d,%d,%d)"%(self.crop_Q_region[0], self.crop_Q_region[1], self.crop_Q_region[2], self.crop_Q_region[3])+"\n")
                else:
                    f.write("crop_Q : "+"False"+"\n")    
                if self.fill:
                    f.write("fill : "+"True"+"\n")
                    f.write("fill_region : "+"(%d,%d,%d,%d)"%(self.fill_region[0], self.fill_region[1], self.fill_region[2], self.fill_region[3])+"\n")
                    f.write("num_neighbor : %d\n"%self.num_neighbor)
                else:
                    f.write("fill : "+"False"+"\n")
                f.write("center : "+"(%f,%f)"%(self.probe_qx0,self.probe_qy0)+"\n")
                f.write("reciprocal_pixel_size : "+"%f"%(semiangle_cutoff*1E-3/self.probe_radius_pixels)+"\n")
                f.write("binsize : %d\n"%self.binsize)
                f.write("semiangle : %f\n"%(semiangle_cutoff*1E-3))
                f.write("defocus : %f\n"%defocus)
                f.write("rotation : %f\n"%rotation_degrees)
                f.write("ptycho_type : "+ptycho_type+"\n")
                f.write("num_iteration : %d\n"%num_iteration)

                if ptycho_type == "singleslice":
                    print("singleslice ptychography")
                elif ptycho_type == "mixed-state":
                    print("mixed-state ptychography")
                    f.write("num_probe : %d\n"%num_probe)
                elif ptycho_type == "multislice":
                    print("multislice ptychography")
                    f.write("num_slice : %d\n"%num_slice)
                    f.write("slice_thickness : %f\n"%slice_thickness)
                elif ptycho_type == "mixed-state-multislice":
                    print("mixed-state-multislice ptychography")
                    f.write("num_probe : %d\n"%num_probe)
                    f.write("num_slice : %d\n"%num_slice)
                    f.write("slice_thickness : %f\n"%slice_thickness)
                else:
                    print("Wrong type!")

                if package == "abtem_latest" or package == "abtem_legacy":
                    f.write("alpha : %f\n"%alpha)
                    f.write("beta : %f\n"%beta)
                    f.write("step_size_damping_rate : %f\n"%step_size_damping_rate)
                    if probe_position_correction:
                        f.write("probe_position_correction : True\n")
                        f.write("pre_position_correction_update_steps : %d\n"%pre_position_correction_update_steps)
                        f.write("position_step_size : %f\n"%position_step_size)
                    else:
                        f.write("probe_position_correction : False\n")

                elif package == "py4dstem":
                    f.write("max_batch_size : %d\n"%max_batch_size)
                    f.write("reconstruction_method : %s\n"%reconstruction_method)
                    f.write("reconstruction_parameter : %f\n"%reconstruction_parameter)
                    f.write("normalization_min : %f\n"%normalization_min)
                    if identical_slices:
                        f.write("identical_slices : True\n")
                    else:
                        f.write("identical_slices : False\n")
                    if object_positivity:
                        f.write("object_positivity : True\n")
                    else:
                        f.write("object_positivity : False\n")

                    if tv_denoise:
                        f.write("tv_denoise : True\n")
                        f.write("tv_denoise_weights : [%f,%f]\n"%(tv_denoise_weights[0], tv_denoise_weights[1]))
                        f.write("tv_denoise_inner_iter : %d\n"%tv_denoise_inner_iter)
                    else:
                        f.write("tv_denoise : False\n")
                        f.write("tv_denoise_weights : None\n")
                        f.write("tv_denoise_inner_iter : None\n")

                    if ptycho_type == "multislice" or ptycho_type == "mixed-state-multislice":
                        if tv_denoise_chambolle:
                            f.write("tv_denoise_chambolle : True\n")
                            f.write("tv_denoise_weight_chambolle : %f\n"%tv_denoise_weight_chambolle)
                        else:
                            f.write("tv_denoise_chambolle : False\n")
                            f.write("tv_denoise_weight_chambolle : None\n")
                else:
                    print("Wrong package!")
                    
        # generate batch file
        if package == "ptyrex":
            with open(save_path+sub_name, 'w') as f:
                f.write("#!/usr/bin/env bash\n")
                f.write("#SBATCH --partition=cs05r\n")
                f.write("#SBATCH --job-name=ptyrex_recon\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=4\n")
                f.write("#SBATCH --cpus-per-task=1\n")
                f.write("#SBATCH --time=12:00:00\n")
                f.write("#SBATCH --output=%s%%j.out\n"%log_path)
                f.write("#SBATCH --error=%s%%j.error\n\n"%log_path)
                if gpu_type == "pascal":
                    f.write("#SBATCH --constraint=NVIDIA_Pascal\n")
                elif gpu_type == "volta":
                    f.write("#SBATCH --constraint=NVIDIA_Volta\n")
                f.write("#SBATCH --gpus-per-node=%d\n"%gpu_node)
                f.write("#SBATCH --mem=0G\n\n")

                f.write("cd /home/ejr78941/ptyrex_temp_5/PtyREX\n")
                f.write("module load python/cuda11.7\n")
                f.write("module load hdf5-plugin/1.12\n")

                f.write("mpirun -np %d ptyrex_recon -c $1"%gpu_node)

        else:
            with open(save_path+sub_name, 'w') as f:
                f.write("#!/usr/bin/env bash\n")
                if device == "gpu":
                    f.write("#SBATCH --partition=cs05r\n")
                    f.write("#SBATCH --gpus-per-node=%d\n"%gpu_node)
                    if gpu_type == "pascal":
                        f.write("#SBATCH --constraint=NVIDIA_Pascal\n")
                    elif gpu_type == "volta":
                        f.write("#SBATCH --constraint=NVIDIA_Volta\n")
                    else:
                        print("Wrong gpu setting!")
                elif device == "cpu":
                    f.write("#SBATCH --partition=cs04r\n")
                else:
                    print("Wrong device!\n")

                if package == "py4dstem":
                    f.write("#SBATCH --job-name=py4dstem_recon\n")
                elif package == "abtem_latest":
                    f.write("#SBATCH --job-name=abtem_latest_recon\n")
                elif package == "abtem_legacy":
                    f.write("#SBATCH --job-name=abtem_legacy_recon\n")
                else:
                    print("Wrong package!\n")

                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --ntasks-per-node=4\n")
                f.write("#SBATCH --cpus-per-task=1\n")
                f.write("#SBATCH --time=48:00:00\n")
                f.write("#SBATCH --mem=0G\n")
                f.write("#SBATCH --output=%s%%j.out\n"%log_path)
                f.write("#SBATCH --error=%s%%j.error\n\n"%log_path)

                if package == "py4dstem":
                    f.write("module load python/3\n")
                    f.write("conda activate /dls/science/groups/e02/Ryu/py_env/python_ptycho\n")
                elif package == "abtem_latest":
                    f.write("module load python/3\n")
                    f.write("conda activate /dls/science/groups/e02/Ryu/py_env/python_ptycho\n")
                elif package == "abtem_legacy":
                    f.write("module load python/3\n")
                    f.write("conda activate /dls/science/groups/e02/Ryu/py_env/abtem_multi\n")
                else:
                    print("Wrong package!\n")

                f.write("python %s %s%s"%(script_path, save_path, cal_name))
                
    def submit_job(self):
        if self.package == "ptyrex":
            sshProcess = subprocess.Popen(['ssh',
                                           '-tt',
                                           'wilson'],
                                           stdin=subprocess.PIPE, 
                                           stdout = subprocess.PIPE,
                                           universal_newlines=True,
                                           bufsize=0)
            sshProcess.stdin.write("echo END\n")
            sshProcess.stdin.write("sbatch "+self.save_path+self.sub_name+' '+self.save_path+self.cal_name+"\n")
            sshProcess.stdin.write("uptime\n")
            sshProcess.stdin.write("logout\n")
            sshProcess.stdin.close()

        else:
            sshProcess = subprocess.Popen(['ssh',
                                           '-tt',
                                           'wilson'],
                                           stdin=subprocess.PIPE, 
                                           stdout = subprocess.PIPE,
                                           universal_newlines=True,
                                           bufsize=0)
            sshProcess.stdin.write("echo END\n")
            sshProcess.stdin.write("sbatch "+self.save_path+self.sub_name+"\n")
            sshProcess.stdin.write("uptime\n")
            sshProcess.stdin.write("logout\n")
            sshProcess.stdin.close()
        
        
def Meta2Config(acc,nCL,aps,):
    if acc == 80e3:
        rot_angle = 238.5
        if aps == 1:
            conv_angle = 41.65
        elif aps == 2:
            conv_angle = 31.74
        elif aps == 3:
            conv_angle = 24.80
        elif aps == 4:
            conv_angle =15.44
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    elif acc == 200e3:
        rot_angle = -77.585
        if aps == 1:
            conv_angle = 37.7
        elif aps == 2:
            conv_angle = 28.8
        elif aps == 3:
            conv_angle = 22.4
        elif aps == 4:
            conv_angle = 14.0
        elif aps == 5:
            conv_angle = 6.4
    elif acc == 300e3:
        rot_angle = -85.5
        if aps == 1:
            conv_angle = 44.7
        elif aps == 2:
            conv_angle = 34.1
        elif aps == 3:
            conv_angle = 26.7
        elif aps == 4:
            conv_angle =16.7
        else:
            print('the aperture being used has unknwon convergence semi angle please consult confluence page or collect calibration data')
    else:
        print('Rotation angle for this acceleration voltage is unknown, please collect calibration data. Rotation angle being set to zero')
        rot_angle = 0

    return rot_angle, conv_angle


def force_fill(img, col_ind, row_ind, num_neighbor=5):
    pad = int(num_neighbor/2)
    padded = np.zeros((img.shape[0]+pad*2, img.shape[1]+pad*2))
    padded[pad:-pad, pad:-pad] = img.copy()

    filled_img = img.copy()

    for col, row in zip(col_ind, row_ind):
        temp = padded[(col):(col+2*pad), (row):(row+2*pad)]
        # calculate the mean of only the surrounding pixels having nonzero values
        if np.nonzero(temp)[0].size != 0:
            avg = (np.sum(temp)-img[col, row]) / np.nonzero(temp)[0].size
        else:
            avg = (np.sum(temp)-img[col, row])
        filled_img[col, row] = avg

    return filled_img