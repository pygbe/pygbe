"""
It contains the necessary functions to set up the surface to be solved.
"""

import time
import numpy
from scipy import linalg

from pygbe.tree.FMMutils import addSources, sortPoints, generateTree, findTwigs
from pygbe.tree.direct import computeDiagonal
from pygbe.util.semi_analytical import GQ_1D
from pygbe.quadrature import quadratureRule_fine
from pygbe.util.readData import (readVertex, readTriangle, readpqr, readcrd,
                                 readFields, read_surface)
from pygbe.classes import Field


def initialize_surface(field_array, filename, param):
    """
    Initialize the surface of the molecule.

    Arguments
    ----------
    field_array: array, contains the Field classes of each region on the surface.
    filename   : name of the file that contains the surface information.
    param      : class, parameters related to the surface.

    Returns
    --------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    """

    surf_array = []

    # Read filenames for surfaces
    files, surf_type, phi0_file = read_surface(filename)
    Nsurf = len(files)

    for i in range(Nsurf):
        print('\nReading surface {} from file {}'.format(i, files[i]))

        s = Surface(Nsurf, surf_type[i], phi0_file[i])
        s.define_surface(files[i], param)
        s.define_regions(field_array, i)
        surf_array.append(s)

    return surf_array


def initialize_field(filename, param):
    """
    Initialize all the regions in the surface to be solved.

    Arguments
    ----------
    filename   : name of the file that contains the surface information.
    param      : class, parameters related to the surface.

    Returns
    --------
    field_array: array, contains the Field classes of each region on the surface.
    """

    LorY, pot, E, kappa, charges, coulomb, qfile, Nparent, parent, Nchild, child = readFields(
        filename)

    LorY = [int(i) if i != 'NA' else 0 for i in LorY]
    E = [complex(i) if 'j' in i else param.REAL(i) if i != 'NA' else 0 for i in E]
    kappa = [param.REAL(i) if i != 'NA' else 0 for i in kappa]
    pot = [int(i) for i in pot]
    coulomb = [int(i) for i in coulomb]
    Nfield = len(LorY)
    field_array = []
    Nchild_aux = 0
    for i in range(Nfield):
        field_aux = Field(LorY[i], kappa[i], E[i], coulomb[i], pot[i])

        if int(charges[i]) == 1:  # if there are charges
            field_aux.load_charges(qfile[i], param.REAL)
        if int(Nparent[i]) == 1:  # if it is an enclosed region
            field_aux.parent.append(int(parent[i]))
            # pointer to parent surface (enclosing surface)
        if int(Nchild[i]) > 0:  # if there are enclosed regions inside
            for j in range(int(Nchild[i])):
                field_aux.child.append(int(child[Nchild_aux + j])
                                       )  # Loop over children to get pointers
            Nchild_aux += int(Nchild[i])  # Point to child for next surface

        field_array.append(field_aux)
    return field_array





