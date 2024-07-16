# Atomic structure analysis with radial (azimuthal) average & variance profiles
# Only compatible with the ePSIC data processig workflow
# Jinseok Ryu (jinseok.ryu@diamond.ac.uk)
# ePSIC, Diamond Light Source
import os
import glob
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import py4DSTEM
import hyperspy.api as hs

try: # when you install 'drca'
    from drca import drca, zero_one_rescale

except: # when you just download the github repository
    import sys
    sys.path.append('/dls/science/groups/e02/Ryu/python_library/drca/drca/')
    from drca import drca, zero_one_rescale

class radial_profile_analysis():

    color_rep = ["black", "red", "green", "blue", "orange", "purple", "yellow", "lime", 
             "cyan", "magenta", "lightgray", "peru", "springgreen", "deepskyblue", 
             "hotpink", "darkgray"]

    cm_rep = ['Reds', 'Greens', 'Blues', 'Oranges', 'Purples']
    
    def __init__(self, base_dir, subfolders, profile_length, num_load, 
                 include_key=None, exclude_key=None, limit_scan_region=[1000, 1000], 
                 simult_edx=False, edx_key='', 
                 roll_axis=True, edx_crop=[0, 0], verbose=True):

        edx_split = []
        edx_avg_split = []
        radial_var_split = []
        radial_var_sum_split = []
        pixel_size_split = []
        loaded_data_path = []

        new_process_flag = True
        
        for i, sub in enumerate(subfolders):
            if simult_edx:
                edx_adrs = glob.glob(base_dir+'/'+sub+'/*/EDX/*.rpl', recursive=True)
                if edx_adrs == []:
                    edx_adrs = glob.glob(base_dir+'/'+sub+'/EDX/*.rpl', recursive=True)
                    if edx_adrs == []:
                        edx_adrs = glob.glob(base_dir+'/'+sub+'EDX/*.rpl', recursive=True)
                        if edx_adrs == []:
                            print("Please make sure that the base directory and subfolder name are correct.")
                            return                

            file_adrs = glob.glob(base_dir+'/'+sub+'/*/*/*.hspy', recursive=True)
            if file_adrs == []:
                file_adrs = glob.glob(base_dir+'/'+sub+'/*/*.hspy', recursive=True)
                if file_adrs == []:
                    file_adrs = glob.glob(base_dir+'/'+sub+'/*.hspy', recursive=True)
                    if file_adrs == []:
                        print("Please make sure that the base directory and subfolder name are correct.")
                        return
            
            file_adrs.sort()
            try:
                edx_adrs.sort()

            except:
                print("No EDX files")
        
            if include_key == []:
                key_list = []
                keyword_ = list(exclude_key)
                keyword_.extend(["mean", "max", "bin"])
                for adr in file_adrs:
                    check = []
                    for key in keyword_:
                        if key in adr:
                            check.append(1)
                    if check == []:
                        key_list.append(adr)
                #print(*key_list, sep="\n")
                print("number of data in subfolder '%s'"%sub)
                print(len(key_list))
                key_list = np.asarray(key_list)

                if simult_edx:
                    edx_adrs = np.asarray(edx_adrs)
                    edx_adrs = edx_adrs[:len(key_list)]
            
                if len(key_list) >= num_load:
                    ri = np.random.choice(len(key_list), num_load, replace=False)
                    file_adr_ = key_list[ri]
                    if simult_edx:
                        edx_adr_ = edx_adrs[ri]
                else:
                    file_adr_ = key_list
                    if simult_edx:
                        edx_adr_ = edx_adrs
        
            else:
                file_adr_ = []
                for adr in file_adrs:
                    for key in include_key:
                        if key in adr and "mean" not in adr:
                            file_adr_.append(adr)
                if simult_edx:
                    edx_adr_ = edx_adrs
                print("number of data in subfolder '%s'"%sub)
                print(len(file_adr_))

            edx_tmp_list = []
            edx_avg_list = []
            radial_var_list = []
            avg_radial_var_list = []
            file_adr = []
            pixel_size_list = []
            for e, adr in enumerate(file_adr_):
                data = hs.load(adr)
                print('original profile size = ', data.data.shape[-1])
                file_adr.append(adr)
                data.data = data.data[:, :, :profile_length]
                local_radial_var_sum = data.mean()
                pixel_size_inv_Ang = local_radial_var_sum.axes_manager[-1].scale

                if simult_edx:
                    edx_data = hs.load(edx_adr_[e]).data
                    if roll_axis:
                        edx_data = np.rollaxis(edx_data, 0, 3)[:limit_scan_region[0], :limit_scan_region[1]]
                    edx_data = hs.signals.Signal1D(edx_data)

                if verbose:
                    print("radial profile data information")
                    print(adr)
                    print(data)
                    print(data.axes_manager)
                    if simult_edx:
                        print("EDX data information")
                        print(edx_adr_[e])
                        print(edx_data)
                        print(edx_data.axes_manager)                   

                radial_var_list.append(data)
                avg_radial_var_list.append(local_radial_var_sum.data)
                pixel_size_list.append(pixel_size_inv_Ang)
                if simult_edx:
                    edx_tmp_list.append(edx_data)
                    edx_avg_list.append(edx_data.mean().data)                    
        
            avg_radial_var_list = np.asarray(avg_radial_var_list)
            radial_var_split.append(radial_var_list)
            radial_var_sum_split.append(avg_radial_var_list)
            pixel_size_split.append(pixel_size_list)
        
            loaded_data_path.append(file_adr)

            if simult_edx:
                edx_avg_list = np.asarray(edx_avg_list)
                edx_split.append(edx_tmp_list)
                edx_avg_split.append(edx_avg_list)                

        # mean profile data load
        loaded_data_mean_path = []
        radial_avg_split = []
        radial_avg_sum_split = []
        for i, sub in enumerate(subfolders):
            radial_avg_list = []
            radial_avg_sum_list = []
            loaded_data_mean = []
            for adr in loaded_data_path[i]:
                dir_path = os.path.dirname(adr)
                data_name = os.path.basename(adr).split("_")
                data_name = data_name[0]+'_'+data_name[1]
                
                try:
                    adr_ = dir_path+"/"+data_name+"_"+'azimuthal_mean.hspy'
                    data = hs.load(adr_)
                    loaded_data_mean.append(adr_)
                except:
                    adr_ = dir_path+"/"+data_name+"_"+'mean.hspy'
                    new_process_flag = False
                    data = hs.load(adr_)
                    loaded_data_mean.append(adr_)
                    
                data.data = data.data[:, :, :profile_length]
                local_radial_avg_sum = data.mean()
                radial_avg_list.append(data)
                radial_avg_sum_list.append(local_radial_avg_sum.data)

            loaded_data_mean_path.append(loaded_data_mean)
        
            radial_avg_split.append(radial_avg_list)
            radial_avg_sum_split.append(radial_avg_sum_list)

            self.radial_avg_split = radial_avg_split
            self.radial_avg_sum_split = radial_avg_sum_split

        # aligned center beam image load
        BF_disc_align = []
        for i, sub in enumerate(subfolders):
            BF_disc_list = []
            for adr in loaded_data_path[i]:
                dir_path = os.path.dirname(adr)
                data_name = os.path.basename(adr).split("_")
                data_name = data_name[0]+'_'+data_name[1]
                
                try:
                    adr_ = dir_path+"/"+data_name+"_"+'azimuthal_data_centre.png'
                    data = plt.imread(adr_)
                except:
                    adr_ = dir_path+"/"+"BF_disc_align_bvm.png"
                    data = plt.imread(adr_)
                    new_process_flag = False
                    
                BF_disc_list.append(data)
        
            BF_disc_align.append(BF_disc_list)


        self.pixel_size_inv_Ang = pixel_size_split[0][0]

        self.base_dir = base_dir
        self.subfolders = subfolders
        self.profile_length = profile_length
        self.num_load = num_load
        self.radial_var_split = radial_var_split
        self.radial_var_sum_split = radial_var_sum_split
        self.pixel_size_split = pixel_size_split
        self.edx_split = edx_split
        self.loaded_data_path = loaded_data_path
        self.loaded_data_mean_path = loaded_data_mean_path
        self.BF_disc_align = BF_disc_align
        self.new_process_flag = new_process_flag
        
        print("data loaded.")


    def center_beam_alignment_check(self, crop=[0, -1, 0, -1]):

        self.crop = crop
        top, bottom, left, right = self.crop
        
        for i in range(len(self.subfolders)):
            num_img = len(self.BF_disc_align[i])
            print(num_img)
            grid_size = int(np.around(np.sqrt(num_img)))
            if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            else:
                fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
            for j in range(num_img):
                ax.flat[j].imshow(self.BF_disc_align[i][j][top:bottom, left:right])
                ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                ax.flat[j].axis("off")
            fig.suptitle(self.subfolders[i]+' BF disc align result')
            fig.tight_layout()
            plt.show()

    
    def intensity_integration_image(self):
    
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_avg_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))
            if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            else:
                fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
            for j in range(num_img):
                sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                ax.flat[j].imshow(sum_map, cmap='inferno')
                ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                ax.flat[j].axis("off")
            fig.suptitle(self.subfolders[i]+' sum of intensities map')
            fig.tight_layout()
            plt.show()

    
    def basic_setup(self, str_path, from_unit, to_unit, broadening=0.01, 
                    fill_width=0.1, height=None, width=None, threshold=None, 
                    distance=None, prominence=0.001):
        print("Original scattering vector range = [%.4f, %.4f]"%(0, self.profile_length*self.pixel_size_inv_Ang))
        
        self.str_path = str_path
        self.from_unit = from_unit
        self.to_unit = to_unit
        
        self.from_ = self.pixel_size_inv_Ang*int(from_unit/self.pixel_size_inv_Ang)
        self.to_ = self.pixel_size_inv_Ang*int(to_unit/self.pixel_size_inv_Ang)
        self.x_axis = np.arange(self.from_, self.to_, self.pixel_size_inv_Ang)
        print("Selected scattering vector range = [%.4f, %.4f]"%(self.x_axis.min(), self.x_axis.max()))
        
        if str_path != []:
            int_sf = {}
            peak_sf = {}
            for adr in self.str_path:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                str_name = adr.split('/')[-1].split('.')[0]
                crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(adr, conventional_standard_structure=True)
                crystal.calculate_structure_factors(self.to_unit)
            
                int_sf[str_name] = py4DSTEM.process.diffraction.utils.calc_1D_profile(
                                            self.x_axis,
                                            crystal.g_vec_leng,
                                            crystal.struct_factors_int,
                                            k_broadening=broadening,
                                            int_scale=1.0,
                                            normalize_intensity=True)

                peaks = find_peaks(int_sf[str_name], 
                                   height=height, 
                                   width=width, 
                                   threshold=threshold, 
                                   distance=distance, 
                                   prominence=prominence)[0]
                
                peaks = peaks * self.pixel_size_inv_Ang
                peaks = peaks + self.from_

                peak_sf[str_name] = peaks
            
                ax.plot(self.x_axis, int_sf[str_name], 'k-', label=str_name)
                ax.legend(loc='right')

                for j, peak in enumerate(peaks):
                    if peak >= self.from_ and peak <= self.to_:
                        print("%d peak position (1/Å):\t"%(j+1), peak)
                        ax.axvline(peak, ls=':', lw=1.5, c='r')
                        ax.fill_between([peak-fill_width, peak+fill_width], y1=np.max(int_sf[str_name]), y2=np.min(int_sf[str_name]), alpha=0.5, color='orange')
                        ax.text(peak, 1.0, "%d"%(j+1))
                
                fig.tight_layout()
                plt.show()                

            int_sf["sum_of_all"] = np.sum(list(int_sf.values()), axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(self.x_axis, int_sf["sum_of_all"], 'k-', label="sum of all")
            ax.legend(loc='right')
            fig.tight_layout()
            plt.show()
    
            self.int_sf = int_sf
            self.peak_sf = peak_sf
    
            print('structure name')
            print(*int_sf.keys(), sep="\n")
    
    
    def sum_radial_profile(self, str_name=None, profile_type="variance"):            

        fig_tot, ax_tot = plt.subplots(2, 1, figsize=(8, 12))
        
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_sum_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))
            
            fig_sub, ax_sub = plt.subplots(2, 1, figsize=(8, 12))
            ax_sub_twin = ax_sub[1].twinx()
        
            total_sp = []
            
            if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size, figsize=(16, 12))
            else:
                fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(16, 10))
            for j, sp in enumerate(self.radial_var_sum_split[i]):
                
                tmp_pixel_size = self.pixel_size_split[i][j]
                from_ = tmp_pixel_size*int(self.from_unit/tmp_pixel_size)
                to_ = tmp_pixel_size*int(self.to_unit/tmp_pixel_size)
                x_axis = np.arange(from_, to_, tmp_pixel_size)

                if profile_type == "variance":
                    tmp_sp = sp[int(from_/tmp_pixel_size):int(to_/tmp_pixel_size)]
                    ax.flat[j].plot(x_axis, tmp_sp, 'k-', label="var_sum")
                elif profile_type == "mean":
                    tmp_sp = self.radial_avg_sum_split[i][j][int(from_/tmp_pixel_size):int(to_/tmp_pixel_size)]
                    ax.flat[j].plot(x_axis, tmp_sp, 'k-', label="mean_sum")
                else:
                    print("Warning! wrong profile type!")
                    return                    
                
                ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                ax.flat[j].legend(loc='upper right')
                
                
                ax_twin = ax.flat[j].twinx()
                if profile_type == "variance":
                    tmp_ap = self.radial_avg_sum_split[i][j][int(from_/tmp_pixel_size):int(to_/tmp_pixel_size)]
                    ax_twin.plot(x_axis, tmp_ap, 'r:', label="mean_sum")
                elif profile_type == "mean":
                    tmp_ap = self.radial_var_sum_split[i][j][int(from_/tmp_pixel_size):int(to_/tmp_pixel_size)]
                    ax_twin.plot(x_axis, tmp_ap, 'r:', label="var_sum")
                else:
                    print("Warning! wrong profile type!")
                    return    
                ax_twin.legend(loc='right')
                
                ax_sub[1].plot(x_axis, tmp_sp/np.max(tmp_sp), label=self.subfolders[i]+"_"+os.path.basename(self.loaded_data_path[i][j])[:15])
                ax_sub[1].set_title("max-normalized")
                ax_sub[0].plot(x_axis, tmp_sp, label=self.subfolders[i]+"_"+os.path.basename(self.loaded_data_path[i][j])[:15])
                ax_sub[0].set_title("without normalization")
                total_sp.append(tmp_sp)

            if str_name != None:
                for key in str_name:
                    ax_sub_twin.plot(self.x_axis, self.int_sf[key], label=key, linestyle=":")
                ax_sub_twin.legend(loc="right")
        
            mean_tot = np.mean(total_sp, axis=0)
            ax_tot[0].plot(x_axis, mean_tot, label=self.subfolders[i])
            ax_tot[0].legend(loc="upper right")
            ax_tot[1].plot(x_axis, mean_tot/np.max(mean_tot), label=self.subfolders[i]+" (max-normalized)")
            ax_tot[1].legend(loc="upper right")
            
            ax_sub[0].legend(loc="upper right")
            ax_sub[1].legend(loc="upper right")
            fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(from_, to_))
            fig.tight_layout()
            if profile_type == "variance":
                fig_sub.suptitle("mean of radial variance profiles - scattering vector range %.3f-%.3f (1/Å)"%(from_, to_))
            else:
                fig_sub.suptitle("mean of radial mean profiles - scattering vector range %.3f-%.3f (1/Å)"%(from_, to_))
            fig_sub.tight_layout()
        if profile_type == "variance":
            fig_tot.suptitle("Compare the mean of radial variance profiles between subfolders")
        else:
            fig_tot.suptitle("Compare the mean of radial mean profiles between subfolders")
        fig_tot.tight_layout()
        plt.show()

    
    def sum_edx(self, edx_from, edx_to, edx_scale=0.01, visual=True):
        self.edx_dim = self.edx_split[0][0].data.shape[2]
        self.edx_range = np.linspace(0.0, self.edx_dim*edx_scale, self.edx_dim)

        self.edx_from = edx_from
        self.edx_to = edx_to
        self.edx_scale = edx_scale

        if visual:
            for i in range(len(self.subfolders)):
                num_img = len(self.edx_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12))
                else:
                    fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10))
                    
                for j in range(num_img):
                    edx_sum_map = np.sum(self.edx_split[i][j].data[:, :, int(edx_from/edx_scale):int(edx_to/edx_scale)], axis=2)
                    ax.flat[j*2].imshow(edx_sum_map, cmap='inferno')
                    ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    ax.flat[j*2].axis("off")
                    
                    edx_sum = np.mean(self.edx_split[i][j].data[:, :, int(edx_from/edx_scale):int(edx_to/edx_scale)], axis=(0, 1))
                    ax.flat[j*2+1].plot(self.edx_range[int(edx_from/edx_scale):int(edx_to/edx_scale)], edx_sum, 'k-')
                fig.suptitle(self.subfolders[i]+' EDX intensity map and mean EDX spectrum')
                fig.tight_layout()
                plt.show()        

    
    def NMF_decompose(self, num_comp, max_normalize=False, rescale_0to1=True, profile_type="variance", verbose=True):
        
        self.num_comp = num_comp
        
        # NMF - load data
        flat_adr = []
        if profile_type == "variance":
            for adrs in self.loaded_data_path:
                flat_adr.extend(adrs)
        elif profile_type == "mean":
            for adrs in self.loaded_data_mean_path:
                flat_adr.extend(adrs)
        else:
            print("Warning! wrong profile type!")
            return
        
        self.run_SI = drca(flat_adr, dat_dim=3, dat_unit='1/Å', cr_range=[self.from_, self.to_, self.pixel_size_inv_Ang], 
                                dat_scale=1, rescale=False, DM_file=True, verbose=verbose)

        # NMF - prepare the input dataset
        self.run_SI.make_input(min_val=0.0, max_normalize=max_normalize, rescale_0to1=rescale_0to1)

        # NMF - NMF decomposition and visualization
        self.run_SI.ini_DR(method="nmf", num_comp=num_comp, result_visual=True, intensity_range="absolute")

    
    def NMF_result(self):
        
        # Loading vectors
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for lv in range(self.num_comp):
            ax.plot(self.x_axis, self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1], label="lv %d"%(lv+1))
        ax.set_facecolor("lightgray")
        ax.legend(loc='upper right')
        fig.tight_layout()
        plt.show()

        # All coefficient maps in one image plot
        if self.num_comp <= len(self.cm_rep):
            num_comp = self.num_comp
            k = 0
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
                for j in range(num_img):
                    sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                    ax.flat[j].imshow(sum_map, cmap='gray')
                    for lv in range(num_comp):
                        ax.flat[j].imshow(self.run_SI.coeffs_reshape[k][:, :, lv], 
                                          cmap=self.cm_rep[lv], 
                                          alpha=self.run_SI.coeffs_reshape[k][:, :, lv]/np.max(self.run_SI.coeffs_reshape[k][:, :, lv]))
                    ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    k += 1
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i])
                fig.tight_layout()
                plt.show()

        else:
            print('The number of loading vectors exceeds the number of the preset colormaps.')
            print(self.cm_rep)
            print('So, it will show the coefficient maps for only 1-%d loading vectors'%(len(self.cm_rep)))
            num_comp = len(self.cm_rep)
            k = 0
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
                for j in range(num_img):
                    sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                    ax.flat[j].imshow(sum_map, cmap='gray')
                    for lv in range(num_comp):
                        ax.flat[j].imshow(self.run_SI.coeffs_reshape[k][:, :, lv], 
                                          cmap=self.cm_rep[lv], 
                                          alpha=self.run_SI.coeffs_reshape[k][:, :, lv]/np.max(self.run_SI.coeffs_reshape[k][:, :, lv]))
                    ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    k += 1
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i])
                fig.tight_layout()
                plt.show()

    
    def NMF_comparison(self, str_name=None, percentile_threshold=90, ref_variance=0.7):
        # Show the pixels with high coefficients for each loading vector and the averaged profiles for the mask region
        coeff_split = []
        thresh_coeff_split = []
        lv_line = []

        for lv in range(self.num_comp):
            lv_tot = np.zeros(self.profile_length)
            total_num = 0
            coeff_lv = []
            thresh_coeff_lv = []
            fig_lv, ax_lv = plt.subplots(1, 3, figsize=(12, 4))
            ax_lv[0].plot(self.x_axis, self.run_SI.DR_comp_vectors[lv], self.color_rep[lv+1])
            ax_twin = ax_lv[0].twinx()
            if str_name != None:
                for key in str_name:
                    ax_twin.plot(self.x_axis, self.int_sf[key], label=key, linestyle=":")
                ax_twin.legend(loc="right")
            ax_lv[0].set_facecolor("lightgray")
            fig_lv.suptitle("Loading vector %d"%(lv+1))
        
            thresh = np.percentile(self.run_SI.DR_coeffs[:, lv], percentile_threshold)
            print("threshold coefficient value for loading vector %d = %f"%(lv+1, thresh))

            k=0
            for i in range(len(self.subfolders)):
                fig_sub_tot, ax_sub_tot = plt.subplots(1, 1, figsize=(8, 6))
                lv_sub = np.zeros(self.profile_length)
                sub_num = 0
                coeff = []
                thresh_coeff = []
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12))
                else:
                    fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10))
                    
                for j in range(num_img):                 
                    coeff_map = self.run_SI.coeffs_reshape[k][:, :, lv].copy()
                    coeff.append(coeff_map)
                    coeff_map[coeff_map<=thresh] = 0
                    coeff_map[coeff_map>thresh] = 1
                    thresh_coeff.append(coeff_map)
                    
                    ax.flat[j*2].imshow(coeff_map, cmap='gray')
                    ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    ax.flat[j*2].axis("off")
                    if len(np.where(coeff_map==1)[0]) != 0:
                        tmp_num = len(np.where(coeff_map==1)[0])
                        sub_num += tmp_num
                        total_num += tmp_num
                        coeff_rv = np.sum(self.radial_var_split[i][j].data[np.where(coeff_map==1)], axis=0)
                        lv_tot += coeff_rv
                        lv_sub += coeff_rv

                        ax.flat[j*2+1].set_ylim(0.0, ref_variance*1.5)
                        ax.flat[j*2+1].plot(self.x_axis, zero_one_rescale(coeff_rv[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)]), 'k-')
                        ax.flat[j*2+1].hlines(y=ref_variance, xmin=self.x_axis.min(), xmax=self.x_axis.max(), color="k", linestyle=":", alpha=0.5)
                        ax.flat[j*2+1].plot(self.x_axis, self.radial_var_sum_split[i][j][int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k:', alpha=0.5)
                        for lva in range(self.num_comp):
                            mean_coeff = np.mean(self.run_SI.coeffs_reshape[k][:, :, lva][np.where(coeff_map==1)])
                            ax.flat[j*2+1].plot(self.x_axis, self.run_SI.DR_comp_vectors[lva]*mean_coeff, self.color_rep[lva+1], alpha=0.7)
                        
                        ax.flat[j*2+1].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                        ax.flat[j*2+1].set_facecolor("lightgray")
                    k+=1
                if sub_num != 0:
                    lv_sub /= sub_num
                ax_sub_tot.plot(self.x_axis, lv_sub[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k-')
                ax_sub_tot.set_title("sum of profiles for all threshold maps - subfolder by subfolder")
                fig_sub_tot.tight_layout()

                ax_lv[2].plot(self.x_axis, lv_sub[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], c=self.color_rep[i], label=self.subfolders[i])
                
                fig.suptitle(self.subfolders[i]+' threshold coefficient map for loading vector %d'%(lv+1))
                fig.tight_layout()
                coeff_lv.append(coeff)
                thresh_coeff_lv.append(thresh_coeff)
                
            coeff_split.append(coeff_lv)
            thresh_coeff_split.append(thresh_coeff_lv)
            
            if total_num != 0:
                lv_tot /= total_num
            lv_line.append(lv_tot)
            ax_lv[1].plot(self.x_axis, lv_tot[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k-')
            ax_lv[1].set_title("sum of profiles for all threshold maps - loading vector %d"%(lv+1))
            ax_lv[2].legend()
            fig_lv.tight_layout()

        fig_tot, ax_tot = plt.subplots(1, 1, figsize=(6, 4))
        for l, line in enumerate(lv_line):
            ax_tot.plot(self.x_axis, line[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], c=self.color_rep[l+1], label='lv %d'%(l+1))
        ax_tot.legend()
        fig_tot.suptitle("Compare the mean of radial profiles between loading vectors")
        fig_tot.tight_layout()
        plt.show()
            
        self.coeff_split = coeff_split
        self.thresh_coeff_split = thresh_coeff_split

        return lv_line


    def scattering_range_of_interest(self, profile_type="variance", str_name=None, fill_width=0.1, height=None, width=None, threshold=None, distance=None, prominence=0.001):

        if width != None:
            width = width/self.pixel_size_inv_Ang

        if distance != None:
            distance = distance/self.pixel_size_inv_Ang
        
        # sum of radial variance profile by subfolder
        total_sum_split = []
        if profile_type == "variance":
            for split in self.radial_var_sum_split:
                total_sum_split.append(np.mean(split, axis=0))
        elif profile_type == "mean":
            for split in self.radial_avg_sum_split:
                total_sum_split.append(np.mean(split, axis=0))  
        else:
            print("Warning! wrong profile type!")
            return

        peak_sub = {}
        for i, sp in enumerate(total_sum_split):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            tmp_sp = sp[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)]
            if np.max(tmp_sp) != 0:
                tmp_sp /= np.max(tmp_sp)

            peaks = find_peaks(tmp_sp, height=height, 
                               width=width, 
                               threshold=threshold, 
                               distance=distance, 
                               prominence=prominence)[0]
            
            peaks = peaks * self.pixel_size_inv_Ang
            peaks = peaks + self.from_
            peak_sub[self.subfolders[i]] = peaks
            
            ax.plot(self.x_axis, tmp_sp, c=self.color_rep[i+1], label=self.subfolders[i])

            if str_name != None:
                ax_twin = ax.twinx()
                for key in str_name:
                    ax_twin.plot(self.x_axis, self.int_sf[key], label=key, linestyle=":")
                ax_twin.legend(loc="right")

            for j, peak in enumerate(peaks):
                if peak >= self.from_ and peak <= self.to_:
                    print("%d peak position (1/Å):\t"%(j+1), peak)
                    ax.axvline(peak, ls=':', lw=1.5, c=self.color_rep[i+1])
                    ax.fill_between([peak-fill_width, peak+fill_width], y1=np.max(tmp_sp), y2=np.min(tmp_sp), alpha=0.5, color='orange')
                    ax.text(peak, 1.0, "%d"%(j+1))

            ax.set_xlabel('scattering vector (1/Å)')
            ax.set_facecolor("lightgray")
            ax.legend()
            fig.tight_layout()
            plt.show()
            
        self.peak_sub = peak_sub

    
    def variance_map(self, sv_range=None, peaks=None, fill_width=0.1):

        if peaks != None:
            for i, peak in enumerate(peaks):
                sv_range = [peak-fill_width, peak+fill_width]
                mean_var_map = []
                std_var_map = []
                for i in range(len(self.subfolders)):
                    num_img = len(self.radial_var_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                    if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                    else:
                        fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
                    for j in range(num_img):
                        var_map = np.sum(self.radial_var_split[i][j].isig[sv_range[0]:sv_range[1]].data, axis=2)
                        mean_var_map.append(np.mean(var_map))
                        std_var_map.append(np.std(var_map))
                        ax.flat[j].imshow(var_map, cmap='inferno')
                        ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    for a in ax.flat:
                        a.axis('off')
                    fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(sv_range[0], sv_range[1]))
                    fig.tight_layout()
                    plt.show()
                
        
                # to specify the absolute threshold value to make the binary variance map
                total_num = 0
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                for i in range(0, len(self.subfolders)):
                    num_img = len(self.radial_var_split[i])
                    if i == 0:
                        num_range = np.arange(0, num_img)
                        ax.plot(num_range, mean_var_map[:num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                        ax.errorbar(num_range, mean_var_map[:num_img], yerr=std_var_map[:num_img], capsize=5, c=self.color_rep[i+1])
                
                    else:
                        num_range = np.arange(total_num, total_num+num_img)
                        ax.plot(num_range, mean_var_map[total_num:total_num+num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                        ax.errorbar(num_range, mean_var_map[total_num:total_num+num_img], 
                                    yerr=std_var_map[total_num:total_num+num_img], capsize=5, c=self.color_rep[i+1])
                    total_num += num_img
                ax.grid()
                ax.legend()
                fig.suptitle("mean and standard deviation of the variance maps above")
                fig.tight_layout()
                plt.show()
            
        elif sv_range != None:
            self.sv_range = sv_range
            mean_var_map = []
            std_var_map = []
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    mean_var_map.append(np.mean(var_map))
                    std_var_map.append(np.std(var_map))
                    ax.flat[j].imshow(var_map, cmap='inferno')
                    ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
                fig.tight_layout()
                plt.show()
            
    
            # to specify the absolute threshold value to make the binary variance map
            total_num = 0
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            for i in range(0, len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                if i == 0:
                    num_range = np.arange(0, num_img)
                    ax.plot(num_range, mean_var_map[:num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                    ax.errorbar(num_range, mean_var_map[:num_img], yerr=std_var_map[:num_img], capsize=5, c=self.color_rep[i+1])
            
                else:
                    num_range = np.arange(total_num, total_num+num_img)
                    ax.plot(num_range, mean_var_map[total_num:total_num+num_img], c=self.color_rep[i+1], marker='s', label=self.subfolders[i])
                    ax.errorbar(num_range, mean_var_map[total_num:total_num+num_img], 
                                yerr=std_var_map[total_num:total_num+num_img], capsize=5, c=self.color_rep[i+1])
                total_num += num_img
            ax.grid()
            ax.legend()
            fig.suptitle("mean and standard deviation of the variance maps above")
            fig.tight_layout()
            plt.show()
        
        else:
            print("The scattering vector range must be specified!")
            return
 

    def high_variance_map(self, abs_threshold=None, peaks=None, fill_width=0.1, percentile_threshold=90):
        # binary variance map (leave only large variances for the range specified above)
        # abosulte variance map threshold (pixel value > abs_threshold will be 1, otherwise it will be 0)
        if peaks != None:
            for p, peak in enumerate(peaks):
                fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6))
                sp_tot = np.zeros(self.profile_length)
                total_num = 0
                sv_range = [peak-fill_width, peak+fill_width]
                
                for i in range(len(self.subfolders)):
                    sp_sub = np.zeros(self.profile_length)
                    sub_num = 0
                    num_img = len(self.radial_var_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                    if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                    else:
                        fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
                    for j in range(num_img):
                        var_map = np.sum(self.radial_var_split[i][j].isig[sv_range[0]:sv_range[1]].data, axis=2)
                        abs_threshold = np.percentile(var_map, percentile_threshold)
                        var_map[var_map<=abs_threshold] = 0
                        var_map[var_map>abs_threshold] = 1
                        ax.flat[j].imshow(var_map, cmap='gray')
                        ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15]+"_threshold value=%f"%abs_threshold)

                        tmp_num = len(np.where(var_map==1)[0])
                        total_num += tmp_num
                        sub_num += tmp_num
                        sp_tot += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                        sp_sub += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)

                    if sub_num != 0:
                        sp_sub /= sub_num
                    ax_tot[1].plot(self.x_axis, sp_sub[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], c=self.color_rep[i], label=self.subfolders[i])
                    
                    for a in ax.flat:
                        a.axis('off')
                    fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(sv_range[0], sv_range[1]))
                    fig.tight_layout()

                ax_tot[1].legend()
                if total_num != 0:
                    sp_tot /= total_num
                ax_tot[0].plot(self.x_axis, sp_tot[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k-')
                fig_tot.tight_layout()
                plt.show()

        elif percentile_threshold != None:     
            thresh_var_split = []
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6))
            sp_tot = np.zeros(self.profile_length)
            total_num = 0

            # to obtain the threshold value
            temp_stored_values = []
            for i in range(len(self.subfolders)):
                num_img = len(self.radial_var_split[i])
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    temp_stored_values.extend(var_map.flatten().tolist())
            self.abs_threshold = np.percentile(temp_stored_values, percentile_threshold)

            # leave the pixels of the high variances
            for i in range(len(self.subfolders)):
                sp_sub = np.zeros(self.profile_length)
                sub_num = 0
                thresh_var = []
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    var_map[var_map<=self.abs_threshold] = 0
                    var_map[var_map>self.abs_threshold] = 1
                    thresh_var.append(var_map)
                    ax.flat[j].imshow(var_map, cmap='gray')
                    ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])

                    tmp_num = len(np.where(var_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    sp_tot += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                    sp_sub += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.x_axis, sp_sub[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], c=self.color_rep[i], label=self.subfolders[i])
                
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
                fig.tight_layout()
                thresh_var_split.append(thresh_var)
                
            ax_tot[1].legend()
            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.x_axis, sp_tot[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k-')
            fig_tot.tight_layout()
            plt.show()
                
            self.thresh_var_split = thresh_var_split
            return sp_tot

        elif abs_threshold != None:     
            thresh_var_split = []
            self.abs_threshold = abs_threshold
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6))
            sp_tot = np.zeros(self.profile_length)
            total_num = 0
            
            for i in range(len(self.subfolders)):
                sp_sub = np.zeros(self.profile_length)
                sub_num = 0
                thresh_var = []
                num_img = len(self.radial_var_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                else:
                    fig, ax = plt.subplots(grid_size, grid_size+1, figsize=(12, 10))
                for j in range(num_img):
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                    var_map[var_map<=self.abs_threshold] = 0
                    var_map[var_map>self.abs_threshold] = 1
                    thresh_var.append(var_map)
                    ax.flat[j].imshow(var_map, cmap='gray')
                    ax.flat[j].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    
                    tmp_num = len(np.where(var_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num                    
                    sp_tot += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                    sp_sub += np.sum(self.radial_var_split[i][j].data[np.where(var_map==1)], axis=0)
                    
                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.x_axis, sp_sub[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], c=self.color_rep[i], label=self.subfolders[i])
                
                for a in ax.flat:
                    a.axis('off')
                fig.suptitle(self.subfolders[i]+' - scattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
                fig.tight_layout()
                thresh_var_split.append(thresh_var)
                
            ax_tot[1].legend()
            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.x_axis, sp_tot[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k-')
            fig_tot.tight_layout()
            plt.show()
                
            self.thresh_var_split = thresh_var_split
            return sp_tot
            
        else:
            print("The absolute threshold value or the percentile threshold must be specified")
            return

    
    def Xcorrel(self, str_name=None, profile_type="mean"):
        xcor_val_split = []
        xcor_sh_split = []

        self.xcor_profile = profile_type
        self.xcor_str = str_name
        
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            xcor_val_list = []
            xcor_sh_list = []
            for j in range(num_img):
                if profile_type=="variance":
                    var_data = self.radial_var_split[i][j].data
                elif profile_type=="mean":
                    var_data = self.radial_avg_split[i][j].data
                else:
                    print("Warning! wrong profile type!")
                    return
                    
                xcor_val = []
                xcor_sh = []
                for sy in range(var_data.shape[0]):
                    for sx in range(var_data.shape[1]):
                        sp = var_data[sy, sx][int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)]
                        xcor = np.correlate(self.int_sf[str_name], sp, mode='full')
                        xcor_val.append(np.max(xcor))
                        xcor_sh.append(np.argmax(xcor))
                xcor_val = np.asarray(xcor_val).reshape(var_data.shape[:2])
                xcor_sh = np.asarray(xcor_sh).reshape(var_data.shape[:2])*self.pixel_size_inv_Ang - np.median(self.x_axis)
                xcor_val_list.append(xcor_val)
                xcor_sh_list.append(xcor_sh)
            xcor_val_split.append(xcor_val_list)
            xcor_sh_split.append(xcor_sh_list)

        self.xcor_val_split = xcor_val_split
        self.xcor_sh_split = xcor_sh_split

        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))
            if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12))
            else:
                fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10))
            for j in range(num_img):     
                ax.flat[j*2].imshow(xcor_val_split[i][j], cmap='inferno')
                ax.flat[j*2].set_title(self.loaded_data_path[i][j][-29:-14])
                ax.flat[j*2].axis("off")
        
                ax.flat[j*2+1].hist(xcor_val_split[i][j].flatten(), bins=100)
                ax.flat[j*2+1].set_title("cross-correlation values")
                ax.flat[j*2+1].set_facecolor("lightgray")
        
            fig.suptitle(self.subfolders[i]+' cross-correlation - value')
            fig.tight_layout()
            plt.show()

    
    def high_Xcorr(self, value_threshold=5.0, shift_threshold=0.3):
        thresh_xcor_split = []
        fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6))
        sp_tot = np.zeros(self.profile_length)
        total_num = 0
        for i in range(len(self.subfolders)):
            sp_sub = np.zeros(self.profile_length)
            thresh_xcor = []
            num_img = len(self.radial_var_split[i])
            grid_size = int(np.around(np.sqrt(num_img)))
            sub_num = 0
            
            if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12))
            else:
                fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10))
            for j in range(num_img):
                xcor_val_map = self.xcor_val_split[i][j].copy()
                xcor_sh_map = self.xcor_sh_split[i][j].copy()
                val_map = self.xcor_val_split[i][j].copy()
                sh_map = self.xcor_sh_split[i][j].copy()        
                xcor_val_map[val_map<=value_threshold] = 0
                xcor_val_map[val_map>value_threshold] = 1
                xcor_sh_map[np.abs(sh_map)>shift_threshold] = 0
                xcor_sh_map[np.abs(sh_map)<=shift_threshold] = 1
        
                bool_mask = xcor_val_map * xcor_sh_map
                thresh_xcor.append(bool_mask)
                ax.flat[j*2].imshow(bool_mask, cmap='gray')
                ax.flat[j*2].set_title(self.loaded_data_path[i][j][-29:-14])
                ax.flat[j*2].axis("off")
                
                if len(np.nonzero(bool_mask)[0]) != 0:
                    if self.xcor_profile == "variance":
                        avg_rv = np.sum(self.radial_var_split[i][j].data[np.where(bool_mask==1)], axis=0)
                        avg_rv = avg_rv[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)]
                    elif self.xcor_profile == "mean":
                        avg_rv = np.sum(self.radial_avg_split[i][j].data[np.where(bool_mask==1)], axis=0)
                        avg_rv = avg_rv[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)]
                    else:
                        print("Warning! wrong profile type!")
                        return

                    tmp_num = len(np.where(bool_mask==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    sp_tot += avg_rv
                    sp_sub += avg_rv
                    if np.max(avg_rv) != 0:
                        avg_rv /= np.max(avg_rv)
                    ax.flat[j*2+1].plot(self.x_axis, avg_rv, 'k-')
                    ax_twin = ax.flat[j*2+1].twinx()
                    ax_twin.plot(self.x_axis, self.int_sf[self.xcor_str], 'k:')
                    ax.flat[j*2+1].set_facecolor("lightgray")

                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.x_axis, sp_sub, c=self.color_rep[i], label=self.subfolders[i])

            thresh_xcor_split.append(thresh_xcor)
            fig.suptitle(self.subfolders[i]+' large cross-correlation - value')
            fig.tight_layout()
        
        ax_tot[1].legend()
        if total_num != 0:
            sp_tot /= total_num
        ax_tot[0].plot(self.x_axis, sp_tot, 'k-')
        fig_tot.tight_layout()
        plt.show()
        self.thresh_xcor_split = thresh_xcor_split
        return sp_tot

    def edx_classification(self, threshold_map="NMF"):
        if threshold_map == "variance":
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6))
            sp_tot = np.zeros_like(self.edx_range)
            total_num = 0
            for i in range(len(self.subfolders)):
                num_img = len(self.edx_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                sub_num = 0
                
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12))
                else:
                    fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10))

                sp_sub = np.zeros_like(self.edx_range)
                for j in range(num_img):
                    thresh_map = self.thresh_var_split[i][j]
                    ax.flat[j*2].imshow(thresh_map, cmap='gray')
                    ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    ax.flat[j*2].axis("off")
                    
                    tmp_num = len(np.where(thresh_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    edx_sum = np.sum(self.edx_split[i][j].data[np.where(thresh_map==1)], axis=0)
                    sp_tot += edx_sum
                    sp_sub += edx_sum
                    ax.flat[j*2+1].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_sub[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], 'k-')

                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_sub[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], c=self.color_rep[i], label=self.subfolders[i])
                fig.suptitle(self.subfolders[i]+' mean EDX spectrum for each high-variance map')
                fig.tight_layout()
                
            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_tot[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], 'k-')
            ax_tot[1].legend()
            fig_tot.suptitle(self.subfolders[i]+' mean EDX spectrum for all high-variance maps')
            fig_tot.tight_layout()
            plt.show()

            return sp_tot

        elif threshold_map == "cross-correlation":
            fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6))
            sp_tot = np.zeros_like(self.edx_range)
            total_num = 0
            for i in range(len(self.subfolders)):
                num_img = len(self.edx_split[i])
                grid_size = int(np.around(np.sqrt(num_img)))
                sub_num = 0
                
                if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                    fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12))
                else:
                    fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10))

                sp_sub = np.zeros_like(self.edx_range)
                for j in range(num_img):
                    thresh_map = self.thresh_xcor_split[i][j]
                    ax.flat[j*2].imshow(thresh_map, cmap='gray')
                    ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                    ax.flat[j*2].axis("off")
                    
                    edx_sum = np.sum(self.edx_split[i][j].data[np.where(thresh_map==1)], axis=0)
                    tmp_num = len(np.where(thresh_map==1)[0])
                    total_num += tmp_num
                    sub_num += tmp_num
                    sp_tot += edx_sum
                    sp_sub += edx_sum
                    ax.flat[j*2+1].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_sub[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], 'k-')

                if sub_num != 0:
                    sp_sub /= sub_num
                ax_tot[1].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_sub[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], c=self.color_rep[i], label=self.subfolders[i])
                fig.suptitle(self.subfolders[i]+' mean EDX spectrum for each high-cross-correlation map')
                fig.tight_layout()

            if total_num != 0:
                sp_tot /= total_num
            ax_tot[0].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_tot[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], 'k-')
            ax_tot[1].legend()
            fig_tot.suptitle(self.subfolders[i]+' mean EDX spectrum for all high-cross-correlation maps')
            fig_tot.tight_layout()
            plt.show()

            return sp_tot
        
        elif threshold_map == "NMF":
            lv_tot = []
            for lv in range(self.num_comp):
                fig_tot, ax_tot = plt.subplots(1, 2, figsize=(16, 6))
                sp_tot = np.zeros_like(self.edx_range)
                total_num = 0
                for i in range(len(self.subfolders)):
                    num_img = len(self.edx_split[i])
                    grid_size = int(np.around(np.sqrt(num_img)))
                    sub_num = 0
                    
                    if (num_img - grid_size**2) <= 0 and (num_img - grid_size**2) > -grid_size:
                        fig, ax = plt.subplots(grid_size, grid_size*2, figsize=(12*2, 12))
                    else:
                        fig, ax = plt.subplots(grid_size, (grid_size+1)*2, figsize=(12*2, 10))
    
                    sp_sub = np.zeros_like(self.edx_range)
                    for j in range(num_img):
                        thresh_map = self.thresh_coeff_split[lv][i][j]
                        ax.flat[j*2].imshow(thresh_map, cmap='gray')
                        ax.flat[j*2].set_title(os.path.basename(self.loaded_data_path[i][j])[:15])
                        ax.flat[j*2].axis("off")
                        
                        edx_sum = np.sum(self.edx_split[i][j].data[np.where(thresh_map==1)], axis=0)
                        tmp_num = len(np.where(thresh_map==1)[0])
                        total_num += tmp_num
                        sub_num += tmp_num
                        sp_tot += edx_sum
                        sp_sub += edx_sum
                        ax.flat[j*2+1].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_sub[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], 'k-')

                    if sub_num != 0:
                        sp_sub /= sub_num
                    ax_tot[1].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_sub[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], c=self.color_rep[i], label=self.subfolders[i])
                    fig.suptitle(self.subfolders[i]+' mean EDX spectrum for each high-coefficient map for loading vector %d'%(lv+1))
                    fig.tight_layout()
                if total_num != 0:
                    sp_tot /= total_num
                lv_tot.append(sp_tot)
                ax_tot[0].plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], sp_tot[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], 'k-')
                ax_tot[1].legend()
                fig_tot.suptitle(self.subfolders[i]+' mean EDX spectrum for all high-coefficient maps for loading vector %d'%(lv+1))
                fig_tot.tight_layout()

            fig_lv, ax_lv = plt.subplots(1, 1, figsize=(12, 4))
            for l, line in enumerate(lv_tot):
                ax_lv.plot(self.edx_range[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], 
                           line[int(self.edx_from/self.edx_scale):int(self.edx_to/self.edx_scale)], c=self.color_rep[l+1], label='lv %d'%(l+1))
            ax_lv.legend()
            fig_lv.suptitle("Compare the mean of EDX spectra between loading vectors")
            fig_lv.tight_layout()
            plt.show()

            return lv_tot
        
        else:
            print("Warning! unavailable type!")

    
    def summary_save(self, sv_range=None, percentile_threshold=None, save=False, obtain_dp=False, log_scale_dp=False):
        
        for i in range(len(self.subfolders)):
            num_img = len(self.radial_var_split[i])
            max_dps = []
            mean_dps = []
            for j in range(num_img):
                save_path = os.path.dirname(self.loaded_data_path[i][j]) # able to change the base directory for saving
                print("save directory: ", save_path)
                data_name = os.path.basename(self.loaded_data_path[i][j]).split("_")
                data_name = data_name[0]+'_'+data_name[1]
                print("save prefix: ", data_name)
                top, bottom, left, right = self.crop
                fig, ax = plt.subplots(3, 3, figsize=(15, 15))
                ax[0, 0].imshow(self.BF_disc_align[i][j][top:bottom, left:right], cmap='inferno')
                ax[0, 0].set_title("Aligned BF disc")
                ax[0, 0].axis("off")

                sum_map = np.sum(self.radial_avg_split[i][j].data, axis=2)
                ax[0, 1].imshow(sum_map, cmap='inferno')
                ax[0, 1].set_title("Intensity map")
                ax[0, 1].axis("off")                 
        
                rv = self.radial_var_sum_split[i][j]
                ax[0, 2].plot(self.x_axis, rv[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k-', label="var_sum")
                ax[0, 2].set_title("Sum of radial variance/mean profiles")
                ax[0, 2].legend(loc='upper right')
                
                ra = self.radial_avg_sum_split[i][j]
                ax_twin = ax[0, 2].twinx()
                ax_twin.plot(self.x_axis, ra[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'r:', label="mean_sum")
                ax_twin.legend(loc='right')

                if sv_range != None:
                    var_map = np.sum(self.radial_var_split[i][j].isig[sv_range[0]:sv_range[1]].data, axis=2)
                else:
                    var_map = np.sum(self.radial_var_split[i][j].isig[self.sv_range[0]:self.sv_range[1]].data, axis=2)
                ax[1, 0].imshow(var_map, cmap='inferno')
                ax[1, 0].set_title('Variance map\nscattering vector range %.3f-%.3f (1/Å)'%(self.sv_range[0], self.sv_range[1]))
        
                th_map = var_map.copy()
                if percentile_threshold != None:
                    abs_threshold = np.percentile(var_map, percentile_threshold)
                    th_map[var_map<=abs_threshold] = 0
                    th_map[var_map>abs_threshold] = 1
                else:
                    th_map[var_map<=self.abs_threshold] = 0
                    th_map[var_map>self.abs_threshold] = 1                    
                ax[1, 1].imshow(th_map, cmap='gray')
                ax[1, 1].set_title('High-variance map\nabsolute threshold %.3f'%self.abs_threshold)

                if obtain_dp and len(np.nonzero(th_map)[0]) != 0:
                    if self.new_process_flag:
                        dataset = hs.load(self.loaded_data_path[i][j][:-18]+'calibrated_and_corrected.zspy')
                    else:      
                        cali = py4DSTEM.read(self.loaded_data_path[i][j][:-13]+"braggdisks_cali.h5")
                        dataset = py4DSTEM.read(self.loaded_data_path[i][j][:-13]+"prepared_data.h5")
                        dataset = py4DSTEM.DataCube(dataset.data)
                        dataset = dataset.filter_hot_pixels(thresh = 0.1, return_mask=False)
                        dataset.calibration = cali.calibration
                        dataset.calibrate()
                        
                    mean_dp = np.mean(dataset.data[np.where(th_map==1)], axis=0)
                    mean_dps.append(np.sum(dataset.data[np.where(th_map==1)], axis=0))
                    if log_scale_dp:
                        mean_dp[mean_dp <= 0] = 1.0
                        ax[2, 1].imshow(np.log(mean_dp), cmap='inferno')
                        ax[2, 1].set_title('(log-scale) Mean diffraction pattern\nfor the high-variance map')
                    else:
                        ax[2, 1].imshow(mean_dp, cmap='inferno')
                        ax[2, 1].set_title('Mean diffraction pattern\nfor the high-variance map')
                        
                    if save:
                        mean_dp = hs.signals.Signal2D(mean_dp)
                        mean_dp.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        mean_dp.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        mean_dp.save(save_path+'/'+data_name+"_mean_diffraction_pattern_for_threshold_map.hspy", overwrite=True)
        
                    max_dp = np.max(dataset.data[np.where(th_map==1)], axis=0)
                    max_dps.append(max_dp)
                    if log_scale_dp:
                        max_dp[max_dp <= 0] = 1.0
                        ax[1, 2].imshow(np.log(max_dp), cmap='inferno')
                        ax[1, 2].set_title('(log-scale) Maximum diffraction pattern\nfor the thresholding map')
                    else:
                        ax[1, 2].imshow(max_dp, cmap='inferno')
                        ax[1, 2].set_title('Maximum diffraction pattern\nfor the high-variance map')
                        
                    if save:
                        max_dp = hs.signals.Signal2D(max_dp)
                        max_dp.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        max_dp.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                        max_dp.save(save_path+'/'+data_name+"_max_diffraction_pattern_for_threshold_map.hspy", overwrite=True)

                    del dataset # release the occupied memory
                    
                if len(np.nonzero(th_map)[0]) != 0:
                    avg_rv = np.mean(self.radial_var_split[i][j].data[np.where(th_map==1)], axis=0)
                    ax[2, 2].plot(self.x_axis, avg_rv[int(self.from_/self.pixel_size_inv_Ang):int(self.to_/self.pixel_size_inv_Ang)], 'k-')
                    ax[2, 2].set_title('Averaged radial variance profile\nfor the high-variance map')
                    if save:
                        avg_rv = hs.signals.Signal1D(avg_rv)
                        avg_rv.axes_manager[0].scale = self.pixel_size_inv_Ang
                        avg_rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile_for_threshold_map.hspy", overwrite=True) 
        
                ax[1, 0].axis("off")
                ax[1, 1].axis("off")
                ax[1, 2].axis("off")
                ax[2, 0].axis("off")
                ax[2, 1].axis("off")
         
                fig.suptitle(self.subfolders[i]+" - "+os.path.basename(self.loaded_data_path[i][j])[:15])
                fig.tight_layout()
        
                if save:
                    sum_map = hs.signals.Signal2D(sum_map)
                    sum_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                    sum_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                    sum_map.save(save_path+'/'+data_name+"_intensity_map.hspy", overwrite=True)
                    rv = hs.signals.Signal1D(rv)
                    rv.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[-1].scale
                    rv.save(save_path+'/'+data_name+"_mean_radial_variance_profile.hspy", overwrite=True)            
                    var_map = hs.signals.Signal2D(var_map)
                    var_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                    var_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                    var_map.save(save_path+'/'+data_name+"_variance_map.hspy", overwrite=True)
                    th_map = hs.signals.Signal2D(th_map)
                    th_map.axes_manager[0].scale = self.radial_var_split[i][j].axes_manager[0].scale
                    th_map.axes_manager[1].scale = self.radial_var_split[i][j].axes_manager[1].scale
                    th_map.save(save_path+'/'+data_name+"_threshold_map.hspy", overwrite=True)
                    fig.savefig(save_path+'/'+data_name+"_summary.png")

            max_dps = np.asarray(max_dps)
            mean_dps = np.asarray(mean_dps)
            fig_dp, ax_dp = plt.subplots(1, 2, figsize=(12, 6))
            if log_scale_dp:
                ax_dp[0].imshow(np.log(np.mean(max_dps, axis=0)), cmap="inferno")
            else:
                ax_dp[0].imshow(np.sum(max_dps, axis=0), cmap="inferno")
            ax_dp[0].axis("off")
            ax_dp[0].set_title("sum of all max diffraction patterns from each data")
            if log_scale_dp:
                ax_dp[1].imshow(np.log(np.mean(mean_dps, axis=0)), cmap="inferno")
            else:
                ax_dp[1].imshow(np.sum(mean_dps, axis=0), cmap="inferno")
            ax_dp[1].axis("off")
            ax_dp[1].set_title("sum of all diffraction patterns")
            fig_dp.tight_layout()
            plt.show()