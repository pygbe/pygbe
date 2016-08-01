"""
It contains the necessary functions to set up the surface to be solved.
"""

import time
import numpy
from scipy import linalg

from pygbe.tree.FMMutils import addSources, sortPoints, generateTree, findTwigs
from pygbe.tree.direct import computeDiagonal
from pygbe.util.semi_analytical import GQ_1D
from pygbe.util.readData import (readVertex, readTriangle, readpqr, readcrd,
                           readFields, read_surface)
from pygbe.quadrature import quadratureRule_fine, getGaussPoints
from pygbe.classes import Surface, Field


def initializeSurf(field_array, filename, param):
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

        s = Surface()

        s.surf_type = surf_type[i]

        if s.surf_type == 'dirichlet_surface' or s.surf_type == 'neumann_surface':
            s.phi0 = numpy.loadtxt(phi0_file[i])
            print('\nReading phi0 file for surface {} from {}'.format(i, phi0_file[i]))

        Area_null = []
        tic = time.time()
        s.vertex = readVertex(files[i] + '.vert', param.REAL)
        triangle_raw = readTriangle(files[i] + '.face', s.surf_type)
        toc = time.time()
        print('Time load mesh: {}'.format(toc - tic))
        Area_null = zeroAreas(s, triangle_raw, Area_null)
        s.triangle = numpy.delete(triangle_raw, Area_null, 0)
        print('Removed areas=0: {}'.format(len(Area_null)))

        # Look for regions inside/outside
        for j in range(Nsurf + 1):
            if len(field_array[j].parent) > 0:
                if field_array[j].parent[0] == i:  # Inside region
                    s.kappa_in = field_array[j].kappa
                    s.Ein = field_array[j].E
                    s.LorY_in = field_array[j].LorY
            if len(field_array[j].child) > 0:
                if i in field_array[j].child:  # Outside region
                    s.kappa_out = field_array[j].kappa
                    s.Eout = field_array[j].E
                    s.LorY_out = field_array[j].LorY

        if s.surf_type != 'dirichlet_surface' and s.surf_type != 'neumann_surface':
            s.E_hat = s.Ein / s.Eout
        else:
            s.E_hat = 1

        surf_array.append(s)
    return surf_array


def zeroAreas(s, triangle_raw, Area_null):
    """
    Looks for "zero-areas", areas that are really small, almost zero. It appends
    them to Area_null list.

    Arguments
    ----------
    s           : class, surface where we whan to look for zero areas.
    triangle_raw: list, triangles of the surface.
    Area_null   : list, contains the zero areas.

    Returns
    --------
    Area_null   : list, indices of the triangles with zero-areas.
    """

    for i in range(len(triangle_raw)):
        L0 = s.vertex[triangle_raw[i, 1]] - s.vertex[triangle_raw[i, 0]]
        L2 = s.vertex[triangle_raw[i, 0]] - s.vertex[triangle_raw[i, 2]]
        normal_aux = numpy.cross(L0, L2)
        Area_aux = linalg.norm(normal_aux) / 2
        if Area_aux < 1e-10:
            Area_null.append(i)

    return Area_null




def initializeField(filename, param):
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

    Nfield = len(LorY)
    field_array = []
    Nchild_aux = 0
    for i in range(Nfield):
        if int(pot[i]) == 1:
            param.E_field.append(i)  # This field is where the energy will be calculated
        field_aux = Field()

        try:
            field_aux.LorY = int(LorY[i])  # Laplace of Yukawa
        except ValueError:
            field_aux.LorY = 0

        if 'j' in E[i]:
            field_aux.E = complex(E[i])
        else:
            try:
                field_aux.E = param.REAL(E[i])  # Dielectric constant
            except ValueError:
                field_aux.E = 0
        try:
            field_aux.kappa = param.REAL(kappa[i])  # inverse Debye length
        except ValueError:
            field_aux.kappa = 0

        field_aux.coulomb = int(coulomb[i])  # do/don't coulomb interaction
        if int(charges[i]) == 1:  # if there are charges
            if qfile[i][-4:] == '.crd':
                xq, q, Nq = readcrd(qfile[i], param.REAL)  # read charges
                print('\nReading crd for region {} from {}'.format(i, qfile[i]))
            if qfile[i][-4:] == '.pqr':
                xq, q, Nq = readpqr(qfile[i], param.REAL)  # read charges
                print('\nReading pqr for region {} from {}'.format(i, qfile[i]))
            field_aux.xq = xq  # charges positions
            field_aux.q = q  # charges values
        if int(Nparent[i]) == 1:  # if it is an enclosed region
            field_aux.parent.append(
                int(parent[i])
            )  # pointer to parent surface (enclosing surface)
        if int(Nchild[i]) > 0:  # if there are enclosed regions inside
            for j in range(int(Nchild[i])):
                field_aux.child.append(int(child[Nchild_aux + j])
                                       )  # Loop over children to get pointers
            Nchild_aux += int(Nchild[i]) - 1  # Point to child for next surface
            Nchild_aux += 1

        field_array.append(field_aux)
    return field_array


def fill_phi(phi, surf_array):
    """
    It places the result vector on surface structure.

    Arguments
    ----------
    phi        : array, result vector.
    surf_array : array, contains the surface classes of each region on the
                        surface.
    """

    s_start = 0
    for i in range(len(surf_array)):
        s_size = len(surf_array[i].xi)
        if surf_array[i].surf_type == 'dirichlet_surface':
            surf_array[i].phi = surf_array[i].phi0
            surf_array[i].dphi = phi[s_start:s_start + s_size]
            s_start += s_size
        elif surf_array[i].surf_type == 'neumann_surface':
            surf_array[i].dphi = surf_array[i].phi0
            surf_array[i].phi = phi[s_start:s_start + s_size]
            s_start += s_size
        elif surf_array[i].surf_type == 'asc_surface':
            surf_array[i].dphi = phi[s_start:s_start + s_size] / surf_array[
                i].Ein
            surf_array[i].phi = numpy.zeros(s_size)
            s_start += s_size
        else:
            surf_array[i].phi = phi[s_start:s_start + s_size]
            surf_array[i].dphi = phi[s_start + s_size:s_start + 2 * s_size]
            s_start += 2 * s_size
