import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

base_path = sys.argv[1]
dir_paths = glob.glob(base_path+"/*")
dir_paths.sort()

for dir_path in dir_paths:
    adrs = glob.glob(dir_path+"/*.hspy", recursive=True)
    hyper_adrs = []
    for adr in adrs:
        if "mean.hspy" in adr:
            hyper_adrs.append(adr)
    for adr in adrs:
        if "var.hspy" in adr:
            hyper_adrs.append(adr)
    for adr in adrs:
        if "scaled.hspy" in adr:
            hyper_adrs.append(adr)  
    print(*hyper_adrs, sep="\n")
    
    if hs.load(hyper_adrs[0]).data.shape[1] == 128:
        print("skip this folder")
        continue
    
    data_list = []
    for adr in hyper_adrs:
        data_list.append(hs.load(adr))
    print(data_list)
    
    data_list[2].data = data_list[2].data.astype('uint8')
    
    intensity_map = np.sum(data_list[0].data[:, :, 0:5], axis=(2))
    print(intensity_map.shape)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(intensity_map, cmap="inferno")
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(dir_path+"/before_scan_correction.png")
    
    mv_points = []
    for i in range(intensity_map.shape[0]):
        mv_points.append(np.argmax(np.abs(np.gradient(intensity_map[i, :]))))

    ref_point = max(set(mv_points), key=mv_points.count)
    print(ref_point)

    for i in range(len(data_list)):
        tmp = data_list[i].data
        corrected = np.zeros(tmp.shape, dtype=data_list[i].data.dtype)
        corrected[:, :tmp.shape[1]-ref_point] = tmp[:, ref_point:]
        corrected[:, tmp.shape[1]-ref_point:] = tmp[:, :ref_point]
        data_list[i].data = corrected
        data_list[i].save(hyper_adrs[i], overwrite=True)
    
    corrected_map = np.sum(data_list[0].data, axis=(2))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(corrected_map, cmap="inferno")
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(dir_path+"/after_scan_correction.png")
    
    del tmp
    del corrected
    del data_list[0]
    del data_list[0]
    del data_list[0]