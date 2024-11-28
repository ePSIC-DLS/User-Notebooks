# Notice
- This guide is only valid if you're using the jupyterhub server of Diamond Light Source
- The ipython notebooks or Python scripts should be modified if you want to perform them on your computer system
- This workflow has been optimised for the Python kernel of 'epsic3.10'  
![Notice](img/jupyterhub_kernel.png)
- [Instructions to connect to Diamond's JupyterHub](https://diamondlightsource.atlassian.net/wiki/spaces/EPSICWEB/pages/167346199/Instructions+to+connect+to+Diamond+s+JupyterHub)
- This guide only deals with the data acquired using E02 (Grand ARM 300; JEOL, MerlinEM; Quantum Detectors, Aztec; Oxford Instruments)

# MIB conversion
- The format of raw 4DSTEM data is '.mib'
- After acquiring the data, the mib files should be converted into the 'hdf5' files using 'MIB_conversion/MIB_convert.ipynb'
- The codes for the MIB conversion can be found in [epsic_tools - MIB_convert_widget](https://github.com/ePSIC-DLS/epsic_tools/tree/master/epsic_tools/mib2hdfConvert/MIB_convert_widget/scripts)
- hdf5 files can be read using 'H5PY' or 'HyperSpy' ([Example](https://github.com/jinseuk56/User-Notebooks/blob/master/ePSIC_Standard_Notebooks/automatic_Au_xgrating_calibration/au_xgrating_cal_submit.ipynb))
- Details can be found inside the notebook  
![MIB_convert](img/mib_conversion.png)
- Currently, when the data is acquired simultaneously with EDX, the scan shape must be manually specified using 'known_shape' widget - the scan shape must be (Scan_X, Scan_Y) = (Scan_X, Scan_X-1)  
![MIB_convert](img/known_shape.png)
# Obtaining the calibration information using the Au reference data
- The reciprocal pixel size of scanning electron nanodiffraction (SEND) data (4DSTEM data acquired using a pencil beam) should be retrieved from the Au reference data
- The ellipticity of diffraction rings should also be calculated
- By running 'automatic_Au_xgrating_calibration/Load_change_submit_array_calibration.ipynb', the calibration information json file can be produced for each reference data  
![calibration](img/au_calibration.png)![calibration](img/au_calibration_json.png)
- Please visit the following page for the details of this process: [py4DSTEM - Au calibration](https://github.com/ePSIC-DLS/Hyperspy_Workshop_2024/blob/main/py4DSTEM/orientation_01_AuAgPd_wire.ipynb)
- The names of the session and the subfolder should be manually entered
- After this process has been finished, it is recommended checking the quality of the process by looking at the notebook file stored in each reference data directory  
![calibration](img/au_calibration_notebook.png)
![calibration](img/au_calibration_result.png)
*Check the calibration quality (comparison between the flattened data and the simulated XRD of gold)*
![calibration](img/ellipticity_correction.png)
*Check the ellipticity calculation (left: ellipticity-corrected, right: original)*
# Transforming the SEND data into the radial (azimuthal) average/variance profile data
- For this, the above processes should be completed (the calibration information will be automatically applied according to the acceleration voltage and camera length)
- This will flatten each 2D diffraction image to a 1D radial average/variance profile
- The exact path of 'automatic_azimuthal_transformation/apply_elliptical_correction_polardatacube.py' should be indicated in 'automatic_azimuthal_transformation/submit_polar_transform_multiple_jobs.ipynb'
- '_azimuthal_mean.hspy' -> average profile 3D data, '_azimuthal_var.hspy' -> variance profile 3D data, '_corrected_scaled.hspy' -> ellpticity-corrected 4DSTEM data  
![transformation](img/radial_files.png)
- Please visit the following page for the details of this process: [py4DSTEM - polar transform](https://github.com/py4dstem/py4DSTEM/blob/dev/py4DSTEM/process/polar/polar_analysis.py)
- This process may be helpful for structure characterisation of amorphous/polycrystalline/mixed-phase materials
# Data analysis for radial profile datasets
- The exact path of 'radial_profile_analysis/radial_profile_analysis.py' should be indicated in 'radial_profile_analysis/Run_radial_profile_analysis.ipynb'
- One or many radial profile datasets can be loaded, and the sum of radial profiles can be seen for each 3D data
- The feature extraction and classification of data can be achieved using a dimensionality reduction method - non-negative matrix factorisation (NMF)
- The EDX spectrum images acquired simultaneously with the SEND data can be also loaded and analysed by correlating with the data analysis result of radial profile datasets  
![EDX](img/EDX_path.png) ![EDX](img/EDX_data.png)  
*The EDX data should be stored as shown in the images above*
- Details can be found inside the notebook
