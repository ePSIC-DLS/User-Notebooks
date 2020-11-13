"""
This file contains function to prepare the xyz file for (py)prismatic simulation
and some utility functions to find d-spacing, angle etc for a general crystal
structure
"""

import numpy as np
import ase.io as io
from ase.visualize import view
from ase import geometry
import ase.build as build

def random_orthogonal_direction(za, a, b, c, alpha, beta, gamma, tol=1e-2, max_ind=5):

    indices = np.zeros(max_ind*2 + 1, dtype=int)
    indices[1::2] = np.arange(1, max_ind+1, dtype=int)
    indices[2::2] = -np.arange(1, max_ind+1, dtype=int)

    candidate = []
    for h2 in indices:
        for k2 in indices:
            for l2 in indices:
                if h2==0 and k2==0 and l2==0:
                    continue
                ang = direction_angle(za, [h2, k2, l2], a, b, c, alpha, beta, gamma)

                if 90-tol < ang < 90+tol:
                    candidate.append(([h2, k2, l2], abs(90-ang)))

    proj = sorted(candidate, key=lambda x: x[1])
    closest = proj[0][0]

    return closest


def get_components(a, b, c, alpha, beta, gamma):

    V = a*b*c*np.sqrt(1 - np.cos(np.deg2rad(alpha))**2 -
                      np.cos(np.deg2rad(beta))**2 - np.cos(np.deg2rad(gamma))**2
                      + 2*np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)))

    S11 = b**2 * c**2 * np.sin(np.deg2rad(alpha))**2
    S22 = a**2 * c**2 * np.sin(np.deg2rad(beta))**2
    S33 = a**2 * b**2 * np.sin(np.deg2rad(gamma))**2
    S12 = a*b*c**2*(np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))-np.cos(np.deg2rad(gamma)))
    S23 = a**2*b*c*(np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma))-np.cos(np.deg2rad(alpha)))
    S13 = a*b**2*c*(np.cos(np.deg2rad(gamma))*np.cos(np.deg2rad(alpha))-np.cos(np.deg2rad(beta)))

    return V, S11, S22, S33, S12, S23, S13

def d_spacing(v, a, b, c, alpha, beta, gamma):
    h, k, l = v

    V, S11, S22, S33, S12, S23, S13 = get_components(a, b, c, alpha, beta, gamma)

    r = (S11*h**2 + S22*k**2 + S33*l**2 + 2*S12*h*k + 2*S23*k*l + 2*S13*h*l) / V**2
    d = np.sqrt(1/r)

    return d

def plane_angle(v1, v2, a, b, c, alpha, beta, gamma):
    h1, k1, l1 = v1
    h2, k2, l2 = v2

    V, S11, S22, S33, S12, S23, S13 = get_components(a, b, c, alpha, beta, gamma)

    d1 = d_spacing(v1, a, b, c, alpha, beta, gamma)
    d2 = d_spacing(v2, a, b, c, alpha, beta, gamma)

    ca = (d1*d2*(S11*h1*h2 + S22*k1*k2 + S33*l1*l2 + S23*(k1*l2+k2*l1) +
                S13*(l1*h2+l2*h1) + S12*(h1*k2+h2*k1))) / V**2

    # fix the rounding error
    if ca > 1:
        ca = 1
    if ca < -1:
        ca = -1

    deg = np.rad2deg(np.arccos(ca))

    return deg



def simplify(v):
    v = np.asarray(v)
    f = np.min(np.abs(v[v!=0]))
    if (np.mod(v/f, 1)==0).all():
        return (v/f).astype(int)
    else:
        return v

def toCartesian(v, a, b, c, alpha, beta, gamma):
    if v is None:
        return None

    v = np.asarray(v)

    M = np.zeros((3,3))
    V = a*b*c*np.sqrt(1 - np.cos(np.deg2rad(alpha))**2 -
                      np.cos(np.deg2rad(beta))**2 - np.cos(np.deg2rad(gamma))**2
                      + 2*np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)))
    M[0,0] = a
    M[0,1] = b*np.cos(np.deg2rad(gamma))
    M[1,1] = b*np.sin(np.deg2rad(gamma))
    M[0,2] = c*np.cos(np.deg2rad(beta))
    M[1,2] = c*(np.cos(np.deg2rad(alpha))-np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)))/np.sin(np.deg2rad(gamma))
    M[2,2] = V/(a*b*np.sin(np.deg2rad(gamma)))

    v_car = M @ v

    return v_car

def toZone(v, a, b, c, alpha, beta, gamma):
    if v is None:
        return None

    v = np.asarray(v)

    M = np.zeros((3,3))
    V = a*b*c*np.sqrt(1 - np.cos(np.deg2rad(alpha))**2 -
                      np.cos(np.deg2rad(beta))**2 - np.cos(np.deg2rad(gamma))**2
                      + 2*np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)))
    M[0,0] = 1/a
    M[0,1] = -np.cos(np.deg2rad(gamma)) / (a*np.sin(np.deg2rad(gamma)))
    M[1,1] = 1 / (b*np.sin(np.deg2rad(gamma)))
    M[0,2] = b*c*(np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(gamma))-np.cos(np.deg2rad(beta))) / (V*np.sin(np.deg2rad(gamma)))
    M[1,2] = a*c*(np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma))-np.cos(np.deg2rad(alpha))) / (V*np.sin(np.deg2rad(gamma)))
    M[2,2] = (a*b*np.sin(np.deg2rad(gamma))) / V

    v_zone = M @ v

    return v_zone


def direction_angle(v1, v2, a, b, c, alpha, beta, gamma):
    p = toCartesian(v1, a, b, c, alpha, beta, gamma)
    q = toCartesian(v2, a, b, c, alpha, beta, gamma)

    ang = np.rad2deg(np.arccos(np.dot(p/np.linalg.norm(p), q/np.linalg.norm(q))))

    return ang


def _parse_crystal_axes(zone_axis, new_x, new_y, a, b, c, alpha, beta, gamma):

    # if Miller-Bravais indices are used for hexagonal system
    if zone_axis is not None and len(zone_axis) == 4:
        zone_axis = hex4to3(zone_axis)
    if new_x is not None and len(new_x) == 4:
        new_x = hex4to3(new_x)
    if new_y is not None and len(new_y) == 4:
        new_y = hex4to3(new_y)

    if new_x is None and new_y is None:
        # raise ValueError('Either the crystal axis along x or y should be provided')
        new_z = np.asarray(zone_axis)
        return new_x, new_y, new_z


    if new_x is not None and new_y is None:
        # new_z and new_x must be orthogonal
        new_x = np.asarray(new_x)
        new_z = np.asarray(zone_axis)
        if not np.isclose(direction_angle(new_x, new_z, a, b, c, alpha, beta, gamma), 90):
            raise ValueError('The provided crystal axes along z and x must be orthogonal')
        new_y = np.cross(new_z, new_x)
        new_y = simplify(new_y)
        return new_x, new_y, new_z

    if new_x is None and new_y is not None:
        # new_z and new_y must be orthogonal
        new_y = np.asarray(new_y)
        new_z = np.asarray(zone_axis)
        if not np.isclose(direction_angle(new_y, new_z, a, b, c, alpha, beta, gamma), 90):
            raise ValueError('The provided crystal axes along z and y must be orthogonal')
        new_x = np.cross(new_y, new_z)
        new_x = simplify(new_x)
        return new_x, new_y, new_z

    if new_x is not None and new_y is not None:
        new_x = np.asarray(new_x)
        new_y = np.asarray(new_y)
        new_z = np.asarray(zone_axis)

        if not np.isclose(direction_angle(new_x, new_y, a, b, c, alpha, beta, gamma), 90) or\
            not np.isclose(direction_angle(new_x, new_z, a, b, c, alpha, beta, gamma), 90) or\
            not np.isclose(direction_angle(new_y, new_z, a, b, c, alpha, beta, gamma), 90):
            raise ValueError('The provided crystal axes must be orthogonal')
        return new_x, new_y, new_z


def hex4to3(hex4):
    U, V, T, W = hex4

    if T != -(U+V):
        raise ValueError('Not a valid Miller-Bravais index')

    u = 2*U + V
    v = U + 2*V
    w = W

    hex3 = simplify([u, v, w])

    return hex3


def orient(atoms, zone_axis, new_x=None, new_y=None):
    ''' Orient an atoms object along a particular zone axis (zone axis along
    the Cartesian z axis)

    Parameters
    ----------
    atoms : Atoms object
        an Atoms object from ASE package, created by reading any file supported
        by ASE
    zone_axis : iterable
        the zone axis, align to the Cartesian z axis (optical axis)
    new_x : iterable, optional
        the new crystal axis along the Cartesian x axis, orthogonal to the zone
        axis and the crystal axis along y axis
    new_y : iterable, optional
        the new crystal axis along the Cartesian y axis, orthogonal to the zone
        axis and the crystal axis along x axis

    Returns
    -------
    rotated : Atoms object
        the oriented Atoms object with the specified crystal axes along x, y, z
    '''

    # define the Cartesian coordinate system
    x = [1,0,0]
    y = [0,1,0]
    z = [0,0,1]

    # don't alter the orginal Atoms object
    rotated = atoms.copy()

    # get information of unit cell
    a, b, c, alpha, beta, gamma = rotated.cell.cellpar()

    # get the crystal axes along the basis of Cartesian system
    new_x, new_y, new_z = _parse_crystal_axes(zone_axis, new_x, new_y,
                                              a, b, c, alpha, beta, gamma)


    print('Crystal axis along x-axis: ', new_x)
    print('Crystal axis along y-axis: ', new_y)
    print('Crystal axis along z-axis: ', new_z)


    new_x_car = toCartesian(new_x, a, b, c, alpha, beta, gamma)
    new_y_car = toCartesian(new_y, a, b, c, alpha, beta, gamma)
    new_z_car = toCartesian(new_z, a, b, c, alpha, beta, gamma)


    # rotate such that the new crystal axes along x and y are aligned with the coordinate system
    if new_x_car is not None and new_y_car is not None:
        build.rotate(rotated, new_x_car, x, new_y_car, y, rotate_cell=True)
    else:
        rotated.rotate(new_z_car, z, rotate_cell=True)

    return rotated



def orthogonalise(atoms, bound=10, thresh=1e-1):
    '''Make the unit cell orthogonal under the current symmetry

    CAUTION: this is a pure python implementation which uses a triple for loop
    to search for linear combination of crystal basis to form orthognal basis,
    increasing search bound above ~50 will take some time and no guarantee that
    it can be found... so find approximation by increasing thresh or do the
    search overnight

    Parameters
    ----------
    atoms : Atoms object
        an Atoms object from ASE package, created by reading any file supported
        by ASE
    bound : int, optional
        the search bound for integer linear combination of crystal bases
    thresh : float, optional
        the tolerance for any non-zero component in the final orthogonal bases.
        The default is 1e-1, in Angstrom.

    Raises
    ------
    ValueError
        when orthogonal bases cannot be found in the search

    Returns
    -------
    superatoms : Atoms object
        the orthongalised supercell with possibly extra atoms
    T : ndarray
        a 3x3 transformation matrix for converting the original unit cell to
        the orthogonal supercell
    '''

    # array defined the original unit cell
    B = atoms.cell[:]

    # define the search range
    srange = np.arange(-bound, bound+1, dtype=int)
    xcan, ycan, zcan = [], [], []

    # perform the search
    for p in srange:
        for q in srange:
            for r in srange:
                vec = p*B[0,:] + q*B[1,:] + r*B[2,:]

                # skip too small vector
                lenvec = np.linalg.norm(vec)
                if lenvec < 1:
                    continue

                # check if the new vector contains two small entries
                # which means it lies either along x, y or z
                mask = np.abs(vec) < thresh
                num_close = np.count_nonzero(mask)

                # record the vector if two of the entries are small
                if num_close == 2:
                    if not mask[0] and vec[0]>0:
                        xcan.append((vec, lenvec, (p,q,r)))
                    elif not mask[1] and vec[1]>0:
                        ycan.append((vec, lenvec, (p,q,r)))
                    elif not mask[2] and vec[2]>0:
                        zcan.append((vec, lenvec, (p,q,r)))

    # if search failed
    if xcan:
        x = sorted(xcan, key=lambda x: x[1])[0]
    else:
        raise ValueError('Vector along x cannot be found')
    if ycan:
        y = sorted(ycan, key=lambda x: x[1])[0]
    else:
        raise ValueError('Vector along y cannot be found')
    if zcan:
        z = sorted(zcan, key=lambda x: x[1])[0]
    else:
        raise ValueError('Vector along z cannot be found')

    # construct the transformation matrix for orthogonalisation
    T = np.empty((3,3), dtype=int)
    T[0,:] = x[2]
    T[1,:] = y[2]
    T[2,:] = z[2]

    # construct the supercell using the found transformation
    superatoms = build.make_supercell(atoms, T, wrap=True, tol=1e-5)

    return superatoms, T



def to_xyz(atoms, outputfile, comment=None, cell_abc=None, occupancy=None,
           rms=0.075, coord_fmt='%.6f'):
    ''' Create a xyz file suitable for Prismatic input from ASE Atoms object

    Parameters
    ----------
    atoms : Atoms object
        an Atoms object from ASE package, created by reading any file supported
        by ASE
    outputfile : str
        the output xyz file name, ideally using full absolute path for
        compatibility with (py)prismatic
    comment : str
        the first line of the xyz file, which is a comment. The default is None.
    cell_abc : iterable
        the lattice constant, better to infer from the Atoms object.
        The default is None.
    occupancy : iterable
        the occupancy of the atoms in the unit cell. The default is None.
    rms : float
        the root mean square displacement due to thermal motion for all the atoms
        (in Angstrom). The default is 0.075.
    coord_fmt : str
        the formatting for the atom coordinate. The default is '%.6f'.
    '''

    # comment
    if comment is None:
        comment = atoms.get_chemical_formula()
    elif not isinstance(comment, str):
        raise ValueError('The comment field needs to be a string')

    # cell dimension
    if cell_abc is None:
        cell_abc = atoms.get_cell_lengths_and_angles()[:3]
    else:
        cell_abc = np.asarray(cell_abc)
        if cell_abc.size != 3:
            raise ValueError('The cell dimension array should have 3 entries')

    # atomic numbers
    atomic_num = atoms.get_atomic_numbers()

    # atom coordinates
    atom_coord = atoms.get_positions()

    # occupancy and RMS displacement
    if occupancy is None:
        occupancy = np.ones(atoms.get_global_number_of_atoms())
    else:
        occupancy = np.asarray(occupancy)
        if occupancy.size != atoms.get_global_number_of_atoms():
            raise ValueError('The size of occupancy array does not match the number of atoms')
    if isinstance(rms, float):
        rms_arr = np.ones(atoms.get_global_number_of_atoms()) * rms
    else:
        raise ValueError('RMS displacement needs to be a float')

    # atom coordinates array
    arr = np.hstack([atomic_num[:,None], atom_coord, occupancy[:,None], rms_arr[:,None]])

    # write prismatic's xyz file
    atom_fmt = ('%d', coord_fmt, coord_fmt, coord_fmt, '%d', coord_fmt)
    with open(outputfile, 'w') as f:
        f.write(comment+'\n')
        np.savetxt(f, np.atleast_2d(cell_abc), delimiter=' ', fmt=coord_fmt)
        np.savetxt(f, arr, delimiter=' ', fmt=atom_fmt)
        f.write('-1\n')

    return