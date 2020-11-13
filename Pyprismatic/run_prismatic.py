"""
This file contains functions to run Prismatic through pyprismatic
"""

import pyprismatic as pr
from pyprismatic.params import Metadata

def electron_wavelength(E0):
    # electron beam energy E0, in keV, return the relativistic wavelength in SI unit
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458
    h = 6.62607e-34

    E0 = E0*1e3
    deno = (2*m*e*E0*(1+e*E0/2/m/c**2))**0.5
    lambda_ = h/deno

    return lambda_

def run_simulation(xyz_file, output_h5, alpha_max=200, write_params=False, **kwargs):
    '''
    alpha_max : float
        the desired maximum scattering angle for the detector, in mrad
    '''

    meta = Metadata(filenameAtoms=xyz_file,
                    filenameOutput=output_h5,
                    potBound=2,
                    **kwargs)

    if kwargs.get('tileX') is None:
        meta.tileX = int(50 / meta.cellDimX)
    if kwargs.get('tileY') is None:
        meta.tileY = int(50 / meta.cellDimY)
    if kwargs.get('tileZ') is None:
        meta.tileZ = int(50 / meta.cellDimZ)

    if kwargs.get('E0') is None:
        meta.E0 = 80 # keV
    if kwargs.get('probeSemiangle') is None:
        meta.probeSemiangle = 30     # mrad
    if kwargs.get('alphaBeamMax') is None:
        meta.alphaBeamMax = 35      # mrad
    if kwargs.get('detectorAngleStep') is None:
        meta.detectorAngleStep = 1   # mrad

    if meta.save2DOutput:
        if kwargs.get('integrationAngleMin') is None:
            meta.integrationAngleMin = 80  # mrad
        if kwargs.get('integrationAngleMax') is None:
            meta.integrationAngleMax = 200 # mrad

    # calculate the pixel size for sampling potential from the desired maximum scattering angle
    alpha_max_SI = (alpha_max+meta.detectorAngleStep) / 1000
    ps = electron_wavelength(meta.E0)*1e10 / 4 / alpha_max_SI
    meta.realspacePixelSizeX = ps
    meta.realspacePixelSizeY = ps

    #meta.toString()
    meta.go()

    if write_params:
        meta.writeParameters('simulation_pyparams.txt')

    return meta
