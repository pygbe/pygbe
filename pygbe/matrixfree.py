"""
Matrix free formulation of the matrix vector product in the GMRES.
"""

import numpy
from numpy import pi

from pygbe.tree.FMMutils import computeIndices, precomputeTerms
from pygbe.tree.direct import coulomb_direct
from pygbe.tree.rhs import calc_aux
from pygbe.projection import project, project_Kt, get_phir, get_phir_gpu
from pygbe.classes import Parameters, IndexConstant
from pygbe.util.semi_analytical import GQ_1D
from datetime import datetime
import os

# PyCUDA libraries
try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
except:
    pass

# Note:
#  Remember ordering of equations:
#      Same order as defined in config file,
#      with internal equation first and then the external equation.


def selfInterior(surf, s, LorY, param, ind0, timing, kernel):
    """
    Self surface interior operator:

    The evaluation point and strengths of the single and double layer potential
    are on the same surface. When we take the limit, the evaluation point
    approaches the surface from the inside, then the result is added to the
    interior equation.

    Arguments
    ----------
    surf  : array, contains all the classes of the surface.
    s     : int, position of the surface in the surface array.
    LorY  : int, Laplace (1) or Yukawa (2).
    param : class, parameters related to the surface.
    ind0  : class, it contains the indices related to the treecode computation.
    timing: class, it contains timing information for different parts of the
                   code.
    kernel: pycuda source module.

    Returns
    --------
    v     : array, result of the matrix-vector product corresponding to the self
            interior interaction.
    """

    K_diag = 2 * pi
    V_diag = 0
    IorE = 1
    
    if numpy.iscomplexobj(surf.XinK) or numpy.iscomplexobj(surf.XinV): 
    #if surf.XinK.dtype == complex or surf.XinV.dtype == complex:

        K_lyr_Re, V_lyr_Re = project(surf.XinK.real, surf.XinV.real, LorY, surf,
                             surf, K_diag, V_diag, IorE, s, param, ind0, timing, kernel)


        K_lyr_Im, V_lyr_Im = project(surf.XinK.imag, surf.XinV.imag, LorY, surf,
                             surf, K_diag, V_diag, IorE, s, param, ind0, timing, kernel)

        K_lyr = K_lyr_Re + 1j * K_lyr_Im
        V_lyr = V_lyr_Re + 1j * V_lyr_Im

    else:
        K_lyr, V_lyr = project(surf.XinK, surf.XinV, LorY, surf, surf, K_diag,
                           V_diag, IorE, s, param, ind0, timing, kernel)

    v = K_lyr - V_lyr

    return v


def selfExterior(surf, s, LorY, param, ind0, timing, kernel):
    """
    Self surface exterior operator:

    The evaluation point and the strengths of both the single and double layer
    potential are on the same surface. When we take the limit, the evaluation
    point approaches the surface from the outside, then the result is added to
    the exterior equation.


    Arguments
    ----------
    surf  : array, contains all the classes of the surface.
    s     : int, position of the surface in the surface array.
    LorY  : int, Laplace (1) or Yukawa (2).
    param : class, parameters related to the surface.
    ind0  : class, it contains the indices related to the treecode computation.
    timing: class, it contains timing information for different parts of the
                   code.
    kernel: pycuda source module.

    Returns
    --------
    v     : array, result of the matrix-vector product corresponding to the self
            exterior interaction.
    K_lyr : array, self exterior double layer potential.
    V_lyr : array, self exterior single layer potential.
    """

    K_diag = -2 * pi
    V_diag = 0.
    IorE = 2

    if numpy.iscomplexobj(surf.XinK) or numpy.iscomplexobj(surf.XinV): 
    #if surf.XinK.dtype == complex or surf.XinV.dtype == complex:

        K_lyr_Re, V_lyr_Re = project(surf.XinK.real, surf.E_hat * surf.XinV.real, LorY, surf,
                             surf, K_diag, V_diag, IorE, s, param, ind0, timing, kernel)

        K_lyr_Im, V_lyr_Im = project(surf.XinK.imag, surf.E_hat * surf.XinV.imag, LorY, surf,
                             surf, K_diag, V_diag, IorE, s, param, ind0, timing, kernel)

        K_lyr = K_lyr_Re + 1j * K_lyr_Im
        V_lyr = V_lyr_Re + 1j * V_lyr_Im

    else:
        K_lyr, V_lyr = project(surf.XinK, surf.E_hat * surf.XinV, LorY, surf, surf, K_diag,
                           V_diag, IorE, s, param, ind0, timing, kernel)


    v = -K_lyr + V_lyr # Note that E_hat is inside the project function call


    return v, K_lyr, V_lyr


def nonselfExterior(surf, src, tar, LorY, param, ind0, timing, kernel):
    """
    Non-self exterior operator:

    The evaluation point resides in a surface that is outside the region
    enclosed by the surface with the strength of the single and double layer
    potentials. If both surfaces share a common external region, we add the
    result to the external equation. If they don't share a common external region
    we add the result to internal equation.


    Arguments
    ----------
    surf  : array, contains all the classes of the surface.
    src   : int, position in the array of the source surface (surface that
                 contains the gauss points).
    tar   : int, position in the array of the target surface (surface that
                 contains the collocation points).
    LorY  : int, Laplace (1) or Yukawa (2).
    param : class, parameters related to the surface.
    ind0  : class, it contains the indices related to the treecode computation.
    timing: class, it contains timing information for different parts of the
                   code.
    kernel: pycuda source module.

    Returns
    --------
    v     : array, result of the matrix-vector product corresponding to the
                   non-self exterior interaction.
    """

    K_diag = 0
    V_diag = 0
    IorE = 1

    if numpy.iscomplexobj(surf[src].XinK) or numpy.iscomplexobj(surf[src].XinV): 
    #if surf[src].XinK.dtype == complex or surf[src].XinV.dtype == complex:

        K_lyr_Re, V_lyr_Re = project(surf[src].XinK.real, surf[src].E_hat*surf[src].XinV.real,
                                 LorY, surf[src], surf[tar], K_diag, V_diag,
                                 IorE, src, param, ind0, timing, kernel)
        K_lyr_Im, V_lyr_Im = project(surf[src].XinK.imag, surf[src].E_hat*surf[src].XinV.imag,
                                 LorY, surf[src], surf[tar], K_diag, V_diag,
                                 IorE, src, param, ind0, timing, kernel)

        K_lyr = K_lyr_Re + 1j * K_lyr_Im
        V_lyr = V_lyr_Re + 1j * V_lyr_Im

    else:
        K_lyr, V_lyr = project(surf[src].XinK, surf[src].E_hat*surf[src].XinV, LorY, surf[src],
                           surf[tar], K_diag, V_diag, IorE, src, param, ind0,
                           timing, kernel)

    v = -K_lyr + V_lyr # Note that E_hat is inside the project function call

    return v


def nonselfInterior(surf, src, tar, LorY, param, ind0, timing, kernel):
    """
    Non-self interior operator:

    The evaluation point resides in a surface that is inside the region
    enclosed by the surface with the strength of the single and double layer
    potentials, and the result has to be added to the exterior equation.

    Arguments
    ----------
    surf  : array, contains all the classes of the surface.
    src   : int, position in the array of the source surface (surface that
                 contains the gauss points).
    tar   : int, position in the array of the target surface (surface that
                 contains the collocation points).
    LorY  : int, Laplace (1) or Yukawa (2).
    param : class, parameters related to the surface.
    ind0  : class, it contains the indices related to the treecode computation.
    timing: class, it contains timing information for different parts of the
                   code.
    kernel: pycuda source module.

    Returns
    --------
    v     : array, result of the matrix-vector product corresponding to the
                   non-self interior interaction.
    """

    K_diag = 0
    V_diag = 0
    IorE = 2

    if numpy.iscomplexobj(surf[src].XinK) or numpy.iscomplexobj(surf[src].XinV): 
    #if surf[src].XinK.dtype == complex or surf[src].XinV.dtype == complex:

        K_lyr_Re, V_lyr_Re = project(surf[src].XinK.real, surf[src].XinV.real,
                                    LorY, surf[src], surf[tar], K_diag, V_diag,
                                    IorE, src, param, ind0, timing, kernel)

        K_lyr_Im, V_lyr_Im = project(surf[src].XinK.imag, surf[src].XinV.imag,
                                    LorY, surf[src], surf[tar], K_diag, V_diag,
                                    IorE, src, param, ind0, timing, kernel)

        K_lyr = K_lyr_Re + 1j * K_lyr_Im
        V_lyr = V_lyr_Re + 1j * V_lyr_Im

    else:

        K_lyr, V_lyr = project(surf[src].XinK, surf[src].XinV, LorY, surf[src],
                           surf[tar], K_diag, V_diag, IorE, src, param, ind0,
                           timing, kernel)
    v = K_lyr - V_lyr

    return v


def selfASC(surf, src, tar, LorY, param, ind0, timing, kernel):
    """
    Self interaction for the aparent surface charge (ASC) formulation.

    Arguments
    ----------
    surf  : array, contains all the classes of the surface.
    src   : int, position in the array of the source surface (surface that
                 contains the gauss points).
    tar   : int, position in the array of the target surface (surface that
                 contains the collocation points).
    LorY  : int, Laplace (1) or Yukawa (2).
    param : class, parameters related to the surface.
    ind0  : class, it contains the indices related to the treecode computation.
    timing: class, it contains timing information for different parts of the
                   code.
    kernel: pycuda source module.

    Returns
    --------
    v     : array, result of the matrix-vector product corresponding to the
                  self interaction in the ASC formulation.
    """

    Kt_diag = -2 * pi * (surf.Eout + surf.Ein) / (surf.Eout - surf.Ein)
    Kt_lyr = project_Kt(surf.XinK, LorY, surf, surf, Kt_diag, src, param, ind0,
                        timing, kernel)

    v = -Kt_lyr

    return v


def gmres_dot(X, surf_array, field_array, ind0, param, timing, kernel):
    """
    It computes the matrix-vector product in the GMRES.

    Arguments
    ----------
    X          : array, initial vector guess.
    surf_array : array, contains the surface classes of each region on the
                        surface.
    field_array: array, contains the Field classes of each region on the
                        surface.
    ind0       : class, it contains the indices related to the treecode
                        computation.
    param      : class, parameters related to the surface.
    timing     : class, it contains timing information for different parts of
                        the code.
    kernel     : pycuda source module.

    Returns
    --------
    MV         : array, resulting matrix-vector multiplication.
    """

    Nfield = len(field_array)
    Nsurf = len(surf_array)

    #   Check if there is a complex dielectric
    if any([numpy.iscomplexobj(f.E) for f in field_array]):
        complex_diel = True
    else:
        complex_diel = False

    #   Place weights on corresponding surfaces and allocate memory
    Naux = 0
    for i in range(Nsurf):
        N = len(surf_array[i].triangle)
        if surf_array[i].surf_type == 'dirichlet_surface':
            if complex_diel:
                surf_array[i].XinK = numpy.zeros(N, dtype=numpy.complex)
            else:
                surf_array[i].XinK = numpy.zeros(N)
            surf_array[i].XinV = X[Naux:Naux + N]
            Naux += N
        elif surf_array[i].surf_type == 'neumann_surface' or surf_array[
                i].surf_type == 'asc_surface':
            surf_array[i].XinK = X[Naux:Naux + N]
            if complex_diel:
                surf_array[i].XinV = numpy.zeros(N, dtype=numpy.complex)
            else:
                surf_array[i].XinV = numpy.zeros(N)
            Naux += N
        else:
            surf_array[i].XinK = X[Naux:Naux + N]
            surf_array[i].XinV = X[Naux + N:Naux + 2 * N]
            Naux += 2 * N

        if complex_diel:
            surf_array[i].Xout_int = numpy.zeros(N, dtype=numpy.complex)
            surf_array[i].Xout_ext = numpy.zeros(N, dtype=numpy.complex)
        else:
            surf_array[i].Xout_int = numpy.zeros(N)
            surf_array[i].Xout_ext = numpy.zeros(N)

#   Loop over fields
    for F in range(Nfield):

        parent_type = 'no_parent'
        if len(field_array[F].parent) > 0:
            parent_type = surf_array[field_array[F].parent[0]].surf_type

        if parent_type == 'asc_surface':
            #           ASC only for self-interaction so far
            LorY = field_array[F].LorY
            p = field_array[F].parent[0]
            v = selfASC(surf_array[p], p, p, LorY, param, ind0, timing, kernel)
            surf_array[p].Xout_int += v

        if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface' and parent_type != 'asc_surface':
            LorY = field_array[F].LorY
            param.kappa = field_array[F].kappa
            if len(field_array[F].parent) > 0:
                p = field_array[F].parent[0]
                v = selfInterior(surf_array[p], p, LorY, param, ind0, timing,
                                 kernel)
                surf_array[p].Xout_int += v

                #           if child surface -> self exterior operator + sibling interaction
                #           sibling interaction: non-self exterior saved on exterior vector
            if len(field_array[F].child) > 0:
                C = field_array[F].child
                for c1 in C:
                    v, t1, t2 = selfExterior(surf_array[c1], c1, LorY, param,
                                             ind0, timing, kernel)
                    surf_array[c1].Xout_ext += v
                    for c2 in C:
                        if c1 != c2:
                            v = nonselfExterior(surf_array, c2, c1, LorY,
                                                param, ind0, timing, kernel)
                            surf_array[c1].Xout_ext += v

#           if child and parent surface -> parent-child and child-parent interaction
#           parent->child: non-self interior saved on exterior vector
#           child->parent: non-self exterior saved on interior vector
            if len(field_array[F].child) > 0 and len(field_array[
                    F].parent) > 0:
                p = field_array[F].parent[0]
                C = field_array[F].child
                for c in C:
                    v = nonselfExterior(surf_array, c, p, LorY, param, ind0,
                                        timing, kernel)
                    surf_array[p].Xout_int += v
                    v = nonselfInterior(surf_array, p, c, LorY, param, ind0,
                                        timing, kernel)
                    surf_array[c].Xout_ext += v

    #   Gather results into the result vector
    if complex_diel:
        MV = numpy.zeros(len(X), dtype=numpy.complex)
    else:
        MV = numpy.zeros(len(X))
    Naux = 0
    for i in range(Nsurf):
        N = len(surf_array[i].triangle)
        if surf_array[i].surf_type == 'dirichlet_surface':
            MV[Naux:Naux + N] = surf_array[i].Xout_ext * surf_array[i].Precond[
                0, :]
            Naux += N
        elif surf_array[i].surf_type == 'neumann_surface':
            MV[Naux:Naux + N] = surf_array[i].Xout_ext * surf_array[i].Precond[
                0, :]
            Naux += N
        elif surf_array[i].surf_type == 'asc_surface':
            MV[Naux:Naux + N] = surf_array[i].Xout_int * surf_array[i].Precond[
                0, :]
            Naux += N
        else:
            MV[Naux:Naux + N] = surf_array[i].Xout_int * surf_array[i].Precond[
                0, :] + surf_array[i].Xout_ext * surf_array[i].Precond[1, :]
            MV[Naux + N:Naux + 2 * N] = surf_array[i].Xout_int * surf_array[
                i].Precond[2, :] + surf_array[i].Xout_ext * surf_array[
                    i].Precond[3, :]
            Naux += 2 * N

    return MV

def locate_s_in_RHS(surf_index, surf_array):
    """Find starting index for current surface on RHS

    This is assembling the RHS of the block matrix. Needs to go through all the
    previous surfaces to find out where on the RHS vector it belongs. If any
    previous surfaces were dirichlet, neumann or asc, then they take up half
    the number of spots in the RHS, so we act accordingly.

    Arguments
    ---------
    surf_index: int, index of surface in question
    surf_array: list, list of all surfaces in problem

    Returns
    -------
    s_start: int, index to insert values on RHS

    """
    s_start = 0
    for surfs in range(surf_index):
        if surf_array[surfs].surf_type in ['dirichlet_surface',
                                           'neumann_surface',
                                           'asc_surface']:
            s_start += len(surf_array[surfs].xi)
        else:
            s_start += 2 * len(surf_array[surfs].xi)

    return s_start

def generateRHS(field_array, surf_array, param, kernel, timing, ind0, electric_field=0):
    """
    It generate the right hand side (RHS) for the GMRES.

    Arguments
    ----------
    field_array: array, contains the Field classes of each region on the surface.
    surf_array : array, contains the surface classes of each region on the
                        surface.
    param      : class, parameters related to the surface.
    kernel     : pycuda source module.
    timing     : class, it contains timing information for different parts of
                        the code.
    ind0       : class, it contains the indices related to the treecode computation.

    Returns
    --------
    F          : array, RHS.
    """

    if any([numpy.iscomplexobj(f.E) for f in field_array]):
        complex_diel = True
    else:
        complex_diel = False

    # Initializing F dtype according to the problem we are solving.
    if complex_diel:
        F = numpy.zeros(param.Neq, dtype=numpy.complex)
    else:
        F = numpy.zeros(param.Neq)

    #   Point charge contribution to RHS
    for field in field_array:
        Nq = len(field.q)
        if Nq > 0:
            #           First look at CHILD surfaces
            for s in field.child:  # Loop over surfaces
                #           Locate position of surface s in RHS
                s_start = locate_s_in_RHS(s, surf_array)
                s_size = len(surf_array[s].xi)
                aux = numpy.zeros_like(surf_array[s].xi)
                stype = 0
                if surf_array[s].surf_type == 'asc_surface':
                    stype = 1
                calc_aux(numpy.ravel(field.q), numpy.ravel(field.xq[:, 0]), numpy.ravel(field.xq[:, 1]), numpy.ravel(field.xq[:, 2]), numpy.ravel(surf_array[s].xi), numpy.ravel(surf_array[s].yi), numpy.ravel(surf_array[s].zi), numpy.ravel(surf_array[s].normal[:, 0]), numpy.ravel(surf_array[s].normal[:, 1]), numpy.ravel(surf_array[s].normal[:, 2]), stype, aux, field.E)
#               For CHILD surfaces, q contributes to RHS in
#               EXTERIOR equation (hence Precond[1,:] and [3,:])

#               No preconditioner
                F[s_start:s_start+s_size] += aux

#               With preconditioner
#               If surface is dirichlet or neumann it has only one equation, affected by Precond[0,:]
#               We do this only here (and not in the parent case) because interaction of charges
#               with dirichlet or neumann surface happens only for the surface as a child surfaces.
#                if surf_array[
#                        s].surf_type == 'dirichlet_surface' or surf_array[
#                            s].surf_type == 'neumann_surface' or surf_array[
#                                s].surf_type == 'asc_surface':
#                    F[s_start:s_start + s_size] += aux * surf_array[s].Precond[
#                        0, :]
#                else:
#                    F[s_start:s_start + s_size] += aux * surf_array[s].Precond[
#                        1, :]
#                    F[s_start + s_size:s_start + 2 *
#                      s_size] += aux * surf_array[s].Precond[3, :]

#           Now look at PARENT surface
            if len(field.parent) > 0:
                #           Locate position of surface s in RHS
                s = field.parent[0]
                s_start = locate_s_in_RHS(s, surf_array)
                s_size = len(surf_array[s].xi)

                aux = numpy.zeros_like(surf_array[s].xi)
                stype = 0
                if surf_array[s].surf_type == 'asc_surface':
                    stype = 1
                calc_aux(numpy.ravel(field.q), numpy.ravel(field.xq[:, 0]), numpy.ravel(field.xq[:, 1]), numpy.ravel(field.xq[:, 2]), numpy.ravel(surf_array[s].xi), numpy.ravel(surf_array[s].yi), numpy.ravel(surf_array[s].zi), numpy.ravel(surf_array[s].normal[:, 0]), numpy.ravel(surf_array[s].normal[:, 1]), numpy.ravel(surf_array[s].normal[:, 2]), stype, aux, field.E)

#               No preconditioner
                F[s_start:s_start+s_size] += aux
#               With preconditioner
#                if surf_array[s].surf_type == 'asc_surface':
#                    F[s_start:s_start + s_size] += aux * surf_array[s].Precond[
#                        0, :]
#                else:
#                    F[s_start:s_start + s_size] += aux * surf_array[s].Precond[
#                        0, :]
#                    F[s_start + s_size:s_start + 2 *
#                      s_size] += aux * surf_array[s].Precond[2, :]


        # Effect of an incomming electric field (only on outmost region)
        # Assuming field comes in z direction
        LorY = field.LorY

        if len(field.parent) == 0 and abs(electric_field) > 1e-12:
             if LorY == 1 and complex_diel == True:
                for s in field.child:  # Loop over child surfaces 
                   #Locate position of surface s in RHS
                   s_start = 0
                   for ss in range(s):
                       if surf_array[
                               ss].surf_type == 'dirichlet_surface' or surf_array[
                                   ss].surf_type == 'neumann_surface' or surf_array[
                                       ss].surf_type == 'asc_surface':
                           print('Surface definition error:')
                           print('Surf type can not be dirichlet, neumann or asc for LSPR problems')

                       else:
                           s_start += 2 * len(surf_array[ss].xi)

                   s_size = len(surf_array[s].xi)

                   tar = surf_array[s]
                   if (tar.surf_type=='dirichlet_surface' or tar.surf_type=='neumann_surface'
                      or tar.surf_type=='asc_surface'):
                       print('LSPR problems required different surface definition')
                       print('Check the input files to correct this')
                       continue

                   else:
                       for s_idx in field.child:
                           src = surf_array[s_idx]   
                           #Assuming field comes in z direction then
                           #electric field contains the - sign in config file
                           phi_field = electric_field*src.normal[:,2]
                           #The contribution is in the exterior equation
                           K_diag = 0
                           V_diag = 0
                           IorE   = 2
                           K_lyr, V_lyr = project(numpy.zeros(len(phi_field)),
                                                   phi_field, LorY, src, tar,
                                                   K_diag, V_diag, IorE, s_idx, param,
                                                   ind0, timing, kernel)
                            
                           # No preconditioner
                           F[s_start:s_start + s_size] += (1 - src.E_hat) * V_lyr 
                           F[s_start+s_size:s_start+2*s_size] += (1 - src.E_hat) * V_lyr 
                           # With preconditioner
                           # F[s_start:s_start + s_size] += (1 - src.E_hat) * V_lyr * tar.Precond[1, :]
                           # F[s_start+s_size:s_start+2*s_size] += (1 - src.E_hat) * V_lyr * tar.Precond[3,:]

             else:
                print("Biomolecule-Surface Under External Electric Field") 

                for s in field.child:
                    param.kappa = field.kappa
                    tar = surf_array[s]
                    for s_idx in field.child:
                        src = surf_array[s_idx]
                        if src.surf_type == 'dielectric_interface' or src.surf_type == 'stern_layer':
                            #Poisson-Boltzmann Equation with Electric Field
                            #Assuming field comes in z direction
                            if LorY == 2 and param.kappa > 1e-12:
                                der_phi_Efield = -electric_field*param.kappa*numpy.exp(-param.kappa*abs(src.zi))
                                phi_Efield = electric_field*numpy.exp(-param.kappa*abs(src.zi))
                            else: 
                                der_phi_Efield = -electric_field*src.normal[:,2]
                                phi_Efield = -electric_field*src.zi

                            K_diag = -2 * pi * (s_idx == s)
                            V_diag = 0
                            IorE = 2

                            K_lyr, V_lyr = project(phi_Efield,
                                                    der_phi_Efield, LorY, src, tar,
                                                    K_diag, V_diag, IorE, s_idx, param,
                                                    ind0, timing, kernel)
                           
                            # Find location of surface s in RHS array
                            s_start = 0
                            for ss in range(s):
                                if surf_array[
                                        ss].surf_type == 'dirichlet_surface' or surf_array[
                                            ss].surf_type == 'neumann_surface' or surf_array[
                                                s].surf_type == 'asc_surface':
                                    s_start += len(surf_array[ss].xi)
                                else:
                                    s_start += 2 * len(surf_array[ss].xi)

                            s_size = len(surf_array[s].xi)

                            if surf_array[
                                    s].surf_type == 'dirichlet_surface' or surf_array[
                                        s].surf_type == 'neumann_surface' or surf_array[
                                            s].surf_type == 'asc_surface':

                                # No preconditioner
                                F[s_start:s_start + s_size] += -K_lyr + V_lyr

                                # With preconditioner
                                # F[s_start:s_start + s_size] += -K_lyr * tar.Precond[0, :] + V_lyr * tar.Precond[0, :] 

                            else:
                                # No preconditioner
                                F[s_start+s_size:s_start+2*s_size] += -K_lyr  + V_lyr 
                                # With preconditioner
                                #F[s_start:s_start + s_size] += -K_lyr * tar.Precond[1, :] + V_lyr * tar.Precond[1, :]

                                #F[s_start+s_size:s_start+2*s_size] += -K_lyr * tar.Precond[3, :] + V_lyr * tar.Precond[3, :]

                        elif src.surf_type == 'dirichlet_surface':
                            if LorY == 2 and param.kappa > 1e-12:
                                der_phi_Efield = numpy.zeros(len(src.zi))
                                phi_Efield = electric_field*numpy.exp(-param.kappa*abs(src.zi))
                            else: 
                                der_phi_Efield = numpy.zeros(len(src.zi))
                                phi_Efield = -electric_field*src.zi                            

                            K_diag_II = -2 * pi * (s_idx == s)
                            V_diag_II = 0
                            IorE = 2
                            K_lyr_EF_II, V_lyr_EF_II = project(phi_Efield,
                                                       der_phi_Efield,
                                                       LorY, src, tar, K_diag_II,
                                                       V_diag_II, IorE, s_idx, param, ind0, timing, kernel)

                            # Find location of surface s in RHS array
                            s_start = 0
                            for ss in range(s):
                                if surf_array[
                                        ss].surf_type == 'dirichlet_surface' or surf_array[
                                            ss].surf_type == 'neumann_surface' or surf_array[
                                                s].surf_type == 'asc_surface':
                                    s_start += len(surf_array[ss].xi)
                                else:
                                    s_start += 2 * len(surf_array[ss].xi)

                            s_size = len(surf_array[s].xi)

                            # No preconditioner
                            if surf_array[
                                    s].surf_type == 'dirichlet_surface' or surf_array[
                                        s].surf_type == 'neumann_surface' or surf_array[
                                            s].surf_type == 'asc_surface':
                                F[s_start:s_start + s_size] += - K_lyr_EF_II 
                            else:
                                F[s_start + s_size:s_start + 2 *
                                  s_size] += - K_lyr_EF_II 

                            
                            # With preconditioner
                            # if s is a charged surface, the surface has only one equation,
                            # else, s has 2 equations and K_lyr affects the external
                            # equation (SIBLING surfaces), which is placed after the internal
                            # one, hence Precond[1,:] and Precond[3,:].
                            #if surf_array[
                            #        s].surf_type == 'dirichlet_surface' or surf_array[
                            #            s].surf_type == 'neumann_surface' or surf_array[
                            #                s].surf_type == 'asc_surface':
                            #    F[s_start:s_start + s_size] += - K_lyr_EF_II * surf_array[
                            #        s].Precond[0, :]
                            #else:
                            #    F[s_start:s_start + s_size] += - K_lyr_EF_II * surf_array[
                            #        s].Precond[1, :]
                            #    F[s_start + s_size:s_start + 2 *
                            #      s_size] += - K_lyr_EF_II * surf_array[
                            #      s].Precond[3, :]

                        elif src.surf_type == 'neumann_surface':
                            if LorY == 2 and param.kappa > 1e-12:
                                der_phi_Efield = -electric_field*param.kappa*numpy.exp(-param.kappa*abs(src.zi))
                                phi_Efield = numpy.zeros(len(src.zi))
                            else: 
                                der_phi_Efield = -electric_field*src.normal[:,2]
                                phi_Efield = numpy.zeros(len(src.zi))

                            K_diag_II = 0
                            V_diag_II = 0
                            IorE = 2
                            K_lyr_EF_II, V_lyr_EF_II = project(phi_Efield,
                                                       der_phi_Efield,
                                                       LorY, src, tar, K_diag_II,
                                                       V_diag_II, IorE, s_idx, param, ind0, timing, kernel)

                            # Find location of surface s in RHS array
                            s_start = 0
                            for ss in range(s):
                                if surf_array[
                                        ss].surf_type == 'dirichlet_surface' or surf_array[
                                            ss].surf_type == 'neumann_surface' or surf_array[
                                                s].surf_type == 'asc_surface':
                                    s_start += len(surf_array[ss].xi)
                                else:
                                    s_start += 2 * len(surf_array[ss].xi)

                            s_size = len(surf_array[s].xi)
                            # No preconditioner
                            if surf_array[
                                    s].surf_type == 'dirichlet_surface' or surf_array[
                                        s].surf_type == 'neumann_surface' or surf_array[
                                            s].surf_type == 'asc_surface':
                                F[s_start:s_start + s_size] += V_lyr_EF_II 
                            else:
                                F[s_start + s_size:s_start + 2 *
                                  s_size] += V_lyr_EF_II 

                            # With preconditioner
                            # if s is a charge surface, the surface has only one equation,
                            # else, s has 2 equations and V_lyr affects the external
                            # equation, which is placed after the internal one, hence
                            # Precond[1,:] and Precond[3,:].
                            #if surf_array[
                            #        s].surf_type == 'dirichlet_surface' or surf_array[
                            #            s].surf_type == 'neumann_surface' or surf_array[
                            #                s].surf_type == 'asc_surface':
                            #    F[s_start:s_start + s_size] += V_lyr_EF_II * surf_array[
                            #        s].Precond[0, :]
                            #else:
                            #    F[s_start:s_start + s_size] += V_lyr_EF_II * surf_array[
                            #        s].Precond[1, :]
                            #    F[s_start + s_size:s_start + 2 *
                            #      s_size] += V_lyr_EF_II * surf_array[s].Precond[3, :]

                        else:
                            continue

#   Dirichlet/Neumann contribution to RHS
#    for field in field_array:

        dirichlet = []
        neumann = []
        LorY = field.LorY

        #       Find Dirichlet and Neumann surfaces in region
        #       Dirichlet/Neumann surfaces can only be child of region,
        #       no point on looking at parent surface
        for s in field.child:
            if surf_array[s].surf_type == 'dirichlet_surface':
                dirichlet.append(s)
            elif surf_array[s].surf_type == 'neumann_surface':
                neumann.append(s)

        if len(neumann) > 0 or len(dirichlet) > 0:

            #           First look at influence on SIBLING surfaces
            for s in field.child:

                param.kappa = field.kappa

                #               Effect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = -2 * pi * (sd == s)
                    V_diag = 0
                    IorE = 2
                    K_lyr, V_lyr = project(surf_array[sd].phi0,
                                           numpy.zeros(len(surf_array[sd].xi)),
                                           LorY, surf_array[sd], surf_array[s],
                                           K_diag, V_diag, IorE, sd, param,
                                           ind0, timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[
                                ss].surf_type == 'dirichlet_surface' or surf_array[
                                    ss].surf_type == 'neumann_surface' or surf_array[
                                        s].surf_type == 'asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2 * len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)

                    # No preconditioner
                    if surf_array[
                            s].surf_type == 'dirichlet_surface' or surf_array[
                                s].surf_type == 'neumann_surface' or surf_array[
                                    s].surf_type == 'asc_surface':
                        F[s_start:s_start + s_size] += K_lyr 
                    else:
                        F[s_start + s_size:s_start + 2 *
                          s_size] += K_lyr 

                    # With preconditioner
                    # if s is a charged surface, the surface has only one equation,
                    # else, s has 2 equations and K_lyr affects the external
                    # equation (SIBLING surfaces), which is placed after the internal
                    # one, hence Precond[1,:] and Precond[3,:].
                    #if surf_array[
                    #        s].surf_type == 'dirichlet_surface' or surf_array[
                    #            s].surf_type == 'neumann_surface' or surf_array[
                    #                s].surf_type == 'asc_surface':
                    #    F[s_start:s_start + s_size] += K_lyr * surf_array[
                    #        s].Precond[0, :]
                    #else:
                    #    F[s_start:s_start + s_size] += K_lyr * surf_array[
                    #        s].Precond[1, :]
                    #    F[s_start + s_size:s_start + 2 *
                    #      s_size] += K_lyr * surf_array[s].Precond[3, :]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE = 2
                    K_lyr, V_lyr = project(
                        numpy.zeros(len(surf_array[sn].phi0)),
                        surf_array[sn].phi0, LorY, surf_array[sn],
                        surf_array[s], K_diag, V_diag, IorE, sn, param, ind0,
                        timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[
                                ss].surf_type == 'dirichlet_surface' or surf_array[
                                    ss].surf_type == 'neumann_surface' or surf_array[
                                        s].surf_type == 'asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2 * len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)
                    
                    # No preconditioner
                    if surf_array[
                            s].surf_type == 'dirichlet_surface' or surf_array[
                                s].surf_type == 'neumann_surface' or surf_array[
                                    s].surf_type == 'asc_surface':
                        F[s_start:s_start + s_size] += -V_lyr
                    else:
                        F[s_start + s_size:s_start + 2 *
                          s_size] += -V_lyr
 
                    # With preconditioner
                    # if s is a charge surface, the surface has only one equation,
                    # else, s has 2 equations and V_lyr affects the external
                    # equation, which is placed after the internal one, hence
                    # Precond[1,:] and Precond[3,:].
                    #if surf_array[
                    #        s].surf_type == 'dirichlet_surface' or surf_array[
                    #            s].surf_type == 'neumann_surface' or surf_array[
                    #                s].surf_type == 'asc_surface':
                    #    F[s_start:s_start + s_size] += -V_lyr * surf_array[
                    #        s].Precond[0, :]
                    #else:
                    #    F[s_start:s_start + s_size] += -V_lyr * surf_array[
                    #        s].Precond[1, :]
                    #    F[s_start + s_size:s_start + 2 *
                    #      s_size] += -V_lyr * surf_array[s].Precond[3, :]

#           Now look at influence on PARENT surface
#           The dirichlet/neumann surface will never be the parent,
#           since we are not solving for anything inside them.
#           Then, in this case we will not look at self interaction,
#           which is dealt with by the sibling surface section
            if len(field.parent) == 1:

                s = field.parent[0]

                #               Effect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = 0
                    V_diag = 0
                    IorE = 2
                    K_lyr, V_lyr = project(surf_array[sd].phi0,
                                           numpy.zeros(len(surf_array[sd].xi)),
                                           LorY, surf_array[sd], surf_array[s],
                                           K_diag, V_diag, IorE, sd, param,
                                           ind0, timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[
                                ss].surf_type == 'dirichlet_surface' or surf_array[
                                    ss].surf_type == 'neumann_surface' or surf_array[
                                        s].surf_type == 'asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2 * len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)
                    
                    # No preconditioner
                    F[s_start:s_start + s_size] += K_lyr 
                    # With preconditioner
                    # Surface s has 2 equations and K_lyr affects the internal
                    # equation, hence Precond[0,:] and Precond[2,:].
                    #F[s_start:s_start + s_size] += K_lyr * surf_array[
                    #    s].Precond[0, :]
                    #F[s_start + s_size:s_start + 2 *
                    #  s_size] += K_lyr * surf_array[s].Precond[2, :]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE = 2
                    K_lyr, V_lyr = project(
                        numpy.zeros(len(surf_array[sn].phi0)),
                        surf_array[sn].phi0, LorY, surf_array[sn],
                        surf_array[s], K_diag, V_diag, IorE, sn, param, ind0,
                        timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = 0
                    for ss in range(s):
                        if surf_array[
                                ss].surf_type == 'dirichlet_surface' or surf_array[
                                    ss].surf_type == 'neumann_surface' or surf_array[
                                        s].surf_type == 'asc_surface':
                            s_start += len(surf_array[ss].xi)
                        else:
                            s_start += 2 * len(surf_array[ss].xi)

                    s_size = len(surf_array[s].xi)
                    
                    # No preconditioner
                    F[s_start:s_start + s_size] += -V_lyr 

                    # With preconditioner
                    # Surface s has 2 equations and K_lyr affects the internal
                    # equation, hence Precond[0,:] and Precond[2,:].
                    #F[s_start:s_start + s_size] += -V_lyr * surf_array[
                    #    s].Precond[0, :]
                    #F[s_start + s_size:s_start + 2 *
                    #  s_size] += -V_lyr * surf_array[s].Precond[2, :]

    return F


def generateRHS_gpu(field_array, surf_array, param, kernel, timing, ind0, electric_field=0):
    """
    It generate the right hand side (RHS) for the GMRES suitable for the GPU.

    Arguments
    ----------
    field_array: array, contains the Field classes of each region on the surface.
    surf_array : array, contains the surface classes of each region on the
                        surface.
    param      : class, parameters related to the surface.
    kernel     : pycuda source module.
    timing     : class, it contains timing information for different parts of
                        the code.
    ind0       : class, it contains the indices related to the treecode
                        computation.

    Returns
    --------
    F          : array, RHS suitable for the GPU.
    """

    if any([numpy.iscomplexobj(f.E) for f in field_array]):
        complex_diel = True
    else: 
        complex_diel = False

    # Initializing F dtype according to the problem we are solving.
    if complex_diel:
        F = numpy.zeros(param.Neq, dtype=numpy.complex)
    else:
        F = numpy.zeros(param.Neq)

    REAL = param.REAL
    computeRHS_gpu = kernel.get_function("compute_RHS")
    computeRHSKt_gpu = kernel.get_function("compute_RHSKt")
    for field in field_array:
        Nq = len(field.q)
        if Nq > 0:

            for s in field.child:  # Loop over surfaces
                surface = surf_array[s]
                s_start = locate_s_in_RHS(s, surf_array)
                s_size = len(surface.xi)
                Nround = len(surface.twig) * param.NCRIT

                GSZ = int(numpy.ceil(float(Nround) / param.NCRIT))  # CUDA grid size

                if surface.surf_type != 'asc_surface':
                    F_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    computeRHS_gpu(F_gpu,
                                   field.xq_gpu,
                                   field.yq_gpu,
                                   field.zq_gpu,
                                   field.q_gpu,
                                   surface.xiDev,
                                   surface.yiDev,
                                   surface.ziDev,
                                   surface.sizeTarDev,
                                   numpy.int32(Nq),
                                   REAL(1),
                                   numpy.int32(param.NCRIT),
                                   numpy.int32(param.BlocksPerTwig),
                                   block=(param.BSZ, 1, 1),
                                   grid=(GSZ, 1))

                    aux = numpy.zeros(Nround)
                    F_gpu.get(aux)
                    aux *= 1./(field.E) #We do this to because if E is
                    #complex, and compute_RHS doesn't accept complex numbers.
                    #so we multiply outside.

                else:
                    Fx_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    Fy_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    Fz_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    computeRHSKt_gpu(Fx_gpu,
                                     Fy_gpu,
                                     Fz_gpu,
                                     field.xq_gpu,
                                     field.yq_gpu,
                                     field.zq_gpu,
                                     field.q_gpu,
                                     surface.xiDev,
                                     surface.yiDev,
                                     surface.ziDev,
                                     surface.sizeTarDev,
                                     numpy.int32(Nq),
                                     REAL(field.E),
                                     numpy.int32(param.NCRIT),
                                     numpy.int32(param.BlocksPerTwig),
                                     block=(param.BSZ, 1, 1),
                                     grid=(GSZ, 1))
                    aux_x = numpy.zeros(Nround)
                    aux_y = numpy.zeros(Nround)
                    aux_z = numpy.zeros(Nround)
                    Fx_gpu.get(aux_x)
                    Fy_gpu.get(aux_y)
                    Fz_gpu.get(aux_z)

                    aux = (aux_x[surface.unsort]*surface.normal[:, 0] +
                           aux_y[surface.unsort]*surface.normal[:, 1] +
                           aux_z[surface.unsort]*surface.normal[:, 2])

                # No preconditioner
                if surface.surf_type in ['dirichlet_surface', 'neumann_surface']:
                    F[s_start:s_start + s_size] += aux[surface.unsort] 
                elif surface.surf_type == 'asc_surface':
                    F[s_start:s_start + s_size] += aux 
                else:
                    F[s_start + s_size:s_start + 2 * s_size] += aux[surface.unsort] 


#               For CHILD surfaces, q contributes to RHS in
#               EXTERIOR equation (hence Precond[1,:] and [3,:])

# With preconditioner
# If surface is dirichlet or neumann it has only one equation, affected by Precond[0,:]
# We do this only here (and not in the parent case) because interaction of charges
# with dirichlet or neumann surface happens only for the surface as a child surfaces.
#                if surface.surf_type in ['dirichlet_surface', 'neumann_surface']:
#                    F[s_start:s_start + s_size] += aux[surface.unsort] * surface.Precond[0, :]
#                elif surface.surf_type == 'asc_surface':
#                    F[s_start:s_start + s_size] += aux * surface.Precond[0, :]
#                else:
#                    F[s_start:s_start + s_size] += aux[surface.unsort] * surface.Precond[1, :]
#                    F[s_start + s_size:s_start + 2 * s_size] += aux[surface.unsort] * surface.Precond[3, :]

#           Now for PARENT surface
            if field.parent:
                s = field.parent[0]
                surface = surf_array[s]

                s_start = locate_s_in_RHS(s, surf_array)
                s_size = len(surface.xi)
                Nround = len(surface.twig) * param.NCRIT

                GSZ = int(numpy.ceil(float(Nround) / param.NCRIT))  # CUDA grid size

                if surface.surf_type != 'asc_surface':
                    F_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    computeRHS_gpu(F_gpu,
                                   field.xq_gpu,
                                   field.yq_gpu,
                                   field.zq_gpu,
                                   field.q_gpu,
                                   surface.xiDev,
                                   surface.yiDev,
                                   surface.ziDev,
                                   surface.sizeTarDev,
                                   numpy.int32(Nq),
                                   REAL(1),
                                   numpy.int32(param.NCRIT),
                                   numpy.int32(param.BlocksPerTwig),
                                   block=(param.BSZ, 1, 1),
                                   grid=(GSZ, 1))

                    aux1 = numpy.zeros(Nround)
                    F_gpu.get(aux1)
                    aux = aux1/(field.E)

                else:
                    Fx_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    Fy_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    Fz_gpu = gpuarray.zeros(Nround, dtype=REAL)
                    computeRHSKt_gpu(Fx_gpu,
                                     Fy_gpu,
                                     Fz_gpu,
                                     field.xq_gpu,
                                     field.yq_gpu,
                                     field.zq_gpu,
                                     field.q_gpu,
                                     surface.xiDev,
                                     surface.yiDev,
                                     surface.ziDev,
                                     surface.sizeTarDev,
                                     numpy.int32(Nq),
                                     REAL(field.E),
                                     numpy.int32(param.NCRIT),
                                     numpy.int32(param.BlocksPerTwig),
                                     block=(param.BSZ, 1, 1),
                                     grid=(GSZ, 1))
                    aux_x = numpy.zeros(Nround)
                    aux_y = numpy.zeros(Nround)
                    aux_z = numpy.zeros(Nround)
                    Fx_gpu.get(aux_x)
                    Fy_gpu.get(aux_y)
                    Fz_gpu.get(aux_z)

                    aux = (aux_x[surface.unsort]*surface.normal[:,0] +
                           aux_y[surface.unsort]*surface.normal[:,1] +
                           aux_z[surface.unsort]*surface.normal[:,2])


#               For PARENT surface, q contributes to RHS in
#               INTERIOR equation (hence Precond[0,:] and [2,:])

#               No preconditioner
                F[s_start:s_start+s_size] += aux[surface.unsort]
#               With preconditioner
#                if surface.surf_type == 'asc_surface':
#                    F[s_start:s_start + s_size] += aux * surface.Precond[
#                        0, :]
#                else:
#                    F[s_start:s_start + s_size] += aux[surface.unsort] * surface.Precond[0, :]
#                    F[s_start + s_size:s_start + 2 * s_size] += aux[surface.unsort] * surface.Precond[2, :]

        # Effect of an incomming electric field (only on outmost region)
        # Assuming field comes in z direction
        LorY = field.LorY

        if len(field.parent) == 0 and abs(electric_field) > 1e-12:
             if LorY == 1 and complex_diel == True:
                for s in field.child:  # Loop over child surfaces
                   #Locate position of surface s in RHS
                   s_start = 0
                   for ss in range(s):
                       if surf_array[
                               ss].surf_type == 'dirichlet_surface' or surf_array[
                                   ss].surf_type == 'neumann_surface' or surf_array[
                                       ss].surf_type == 'asc_surface':
                           print('Surface definition error:')
                           print('Surf type can not be dirichlet, neumann or asc for LSPR problems')

                       else:
                           s_start += 2 * len(surf_array[ss].xi)

                   s_size = len(surf_array[s].xi)

                   tar = surf_array[s]
                   if (tar.surf_type=='dirichlet_surface' or tar.surf_type=='neumann_surface'
                      or tar.surf_type=='asc_surface'):
                       print('LSPR problems required different surface definition')
                       print('Check the input files to correct this')
                       continue

                   else:
                       for s_idx in field.child:
                           src = surf_array[s_idx]
                           #Assuming field comes in z direction then
                           #electric field contains - sign in config file
                           phi_field = electric_field*src.normal[:,2]
                           #The contribution is in the exterior equation
                           K_diag = 0
                           V_diag = 0
                           IorE   = 2

                           K_lyr, V_lyr = project(numpy.zeros(len(phi_field)),
                                               phi_field, LorY, src, tar,
                                               K_diag, V_diag, IorE, s_idx, param,
                                               ind0, timing, kernel)

                           F[s_start+s_size:s_start+2*s_size] += (1 - src.E_hat) * V_lyr
                            # With preconditioner
#                           F[s_start:s_start + s_size] += (1 - src.E_hat) * V_lyr * tar.Precond[1, :]
#                           F[s_start+s_size:s_start+2*s_size] += (1 - src.E_hat) * V_lyr * tar.Precond[3,:]

             else:
                print("Biomolecule-Surface Under External Electric Field") 

                for s in field.child:
                    param.kappa = field.kappa
                    tar = surf_array[s]
                    for s_idx in field.child:
                        src = surf_array[s_idx]
                        if src.surf_type == 'dielectric_interface' or src.surf_type == 'stern_layer':
                            #Poisson-Boltzmann Equation with Electric Field
                            #Assuming field comes in z direction
                            if LorY == 2 and param.kappa > 1e-12:
                                der_phi_Efield = -electric_field*param.kappa*numpy.exp(-param.kappa*abs(src.zi))
                                phi_Efield = electric_field*numpy.exp(-param.kappa*abs(src.zi))
                            else: 
                                der_phi_Efield = -electric_field*src.normal[:,2]
                                phi_Efield = -electric_field*src.zi 

                            K_diag = -2 * pi * (s_idx == s)
                            V_diag = 0
                            IorE = 2

                            K_lyr, V_lyr = project(phi_Efield,
                                                    der_phi_Efield, LorY, src, tar,
                                                    K_diag, V_diag, IorE, s_idx, param,
                                                    ind0, timing, kernel)
                           
                            # Find location of surface s in RHS array
                            s_start = 0
                            for ss in range(s):
                                if surf_array[
                                        ss].surf_type == 'dirichlet_surface' or surf_array[
                                            ss].surf_type == 'neumann_surface' or surf_array[
                                                s].surf_type == 'asc_surface':
                                    s_start += len(surf_array[ss].xi)
                                else:
                                    s_start += 2 * len(surf_array[ss].xi)

                            s_size = len(surf_array[s].xi)

                            # No preconditioner
                            if surf_array[
                                    s].surf_type == 'dirichlet_surface' or surf_array[
                                        s].surf_type == 'neumann_surface' or surf_array[
                                            s].surf_type == 'asc_surface':

                                F[s_start:s_start + s_size] += -K_lyr + V_lyr

                            else:
                                F[s_start+s_size:s_start+2*s_size] += -K_lyr + V_lyr 

                            # With preconditioner
#                            if surf_array[
#                                    s].surf_type == 'dirichlet_surface' or surf_array[
#                                        s].surf_type == 'neumann_surface' or surf_array[
#                                            s].surf_type == 'asc_surface':
#
#                               F[s_start:s_start + s_size] += -K_lyr * tar.Precond[0, :] + V_lyr * tar.Precond[0, :] 

#                            else:
#                                F[s_start:s_start + s_size] += -K_lyr * tar.Precond[1, :] + V_lyr * tar.Precond[1, :]
#
#                                F[s_start+s_size:s_start+2*s_size] += -K_lyr * tar.Precond[3, :] + V_lyr * tar.Precond[3, :]

                        elif src.surf_type == 'dirichlet_surface':
                            if LorY == 2 and param.kappa > 1e-12:
                                der_phi_Efield = numpy.zeros(len(src.zi))
                                phi_Efield = electric_field*numpy.exp(-param.kappa*abs(src.zi))
                            else: 
                                der_phi_Efield = numpy.zeros(len(src.zi))
                                phi_Efield = -electric_field*src.zi                    

                            K_diag_II = -2 * pi * (s_idx == s)
                            V_diag_II = 0
                            IorE = 2
                            K_lyr_EF_II, V_lyr_EF_II = project(phi_Efield,
                                                       der_phi_Efield,
                                                       LorY, src, tar, K_diag_II,
                                                       V_diag_II, IorE, s_idx, param, ind0, timing, kernel)

                            # Find location of surface s in RHS array
                            s_start = 0
                            for ss in range(s):
                                if surf_array[
                                        ss].surf_type == 'dirichlet_surface' or surf_array[
                                            ss].surf_type == 'neumann_surface' or surf_array[
                                                s].surf_type == 'asc_surface':
                                    s_start += len(surf_array[ss].xi)
                                else:
                                    s_start += 2 * len(surf_array[ss].xi)

                            s_size = len(surf_array[s].xi)

                            # No preconditioner
                            if surf_array[
                                    s].surf_type == 'dirichlet_surface' or surf_array[
                                        s].surf_type == 'neumann_surface' or surf_array[
                                            s].surf_type == 'asc_surface':
                                F[s_start:s_start + s_size] += - K_lyr_EF_II 
                            else:
                                F[s_start + s_size:s_start + 2 *
                                  s_size] += - K_lyr_EF_II


                            # With preconditioner
                            # if s is a charged surface, the surface has only one equation,
                            # else, s has 2 equations and K_lyr affects the external
                            # equation (SIBLING surfaces), which is placed after the internal
                            # one, hence Precond[1,:] and Precond[3,:].
                            #if surf_array[
                            #        s].surf_type == 'dirichlet_surface' or surf_array[
                            #            s].surf_type == 'neumann_surface' or surf_array[
                            #                s].surf_type == 'asc_surface':
                            #    F[s_start:s_start + s_size] += - K_lyr_EF_II * surf_array[
                            #        s].Precond[0, :]
                            #else:
                            #    F[s_start:s_start + s_size] += - K_lyr_EF_II * surf_array[
                            #        s].Precond[1, :]
                            #    F[s_start + s_size:s_start + 2 *
                            #      s_size] += - K_lyr_EF_II * surf_array[
                            #      s].Precond[3, :]

                        elif src.surf_type == 'neumann_surface':
                            if LorY == 2 and param.kappa > 1e-12:
                                der_phi_Efield = -electric_field*param.kappa*numpy.exp(-param.kappa*abs(src.zi))
                                phi_Efield = numpy.zeros(len(src.zi))
                            else: 
                                der_phi_Efield = -electric_field*src.normal[:,2]
                                phi_Efield = numpy.zeros(len(src.zi))

                            K_diag_II = 0
                            V_diag_II = 0
                            IorE = 2
                            K_lyr_EF_II, V_lyr_EF_II = project(phi_Efield,
                                                       der_phi_Efield,
                                                       LorY, src, tar, K_diag_II,
                                                       V_diag_II, IorE, s_idx, param, ind0, timing, kernel)

                            # Find location of surface s in RHS array
                            s_start = 0
                            for ss in range(s):
                                if surf_array[
                                        ss].surf_type == 'dirichlet_surface' or surf_array[
                                            ss].surf_type == 'neumann_surface' or surf_array[
                                                s].surf_type == 'asc_surface':
                                    s_start += len(surf_array[ss].xi)
                                else:
                                    s_start += 2 * len(surf_array[ss].xi)

                            s_size = len(surf_array[s].xi)

                            # No preconditioner
                            if surf_array[
                                    s].surf_type == 'dirichlet_surface' or surf_array[
                                        s].surf_type == 'neumann_surface' or surf_array[
                                            s].surf_type == 'asc_surface':
                                F[s_start:s_start + s_size] += V_lyr_EF_II 
                            else:
                                F[s_start + s_size:s_start + 2 *
                                  s_size] += V_lyr_EF_II 
 
                            # With preconditioner
                            # if s is a charge surface, the surface has only one equation,
                            # else, s has 2 equations and V_lyr affects the external
                            # equation, which is placed after the internal one, hence
                            # Precond[1,:] and Precond[3,:].
                            #if surf_array[
                            #        s].surf_type == 'dirichlet_surface' or surf_array[
                            #            s].surf_type == 'neumann_surface' or surf_array[
                            #                s].surf_type == 'asc_surface':
                            #    F[s_start:s_start + s_size] += V_lyr_EF_II * surf_array[
                            #        s].Precond[0, :]
                            #else:
                            #    F[s_start:s_start + s_size] += V_lyr_EF_II * surf_array[
                            #        s].Precond[1, :]
                            #    F[s_start + s_size:s_start + 2 *
                            #      s_size] += V_lyr_EF_II * surf_array[s].Precond[3, :]

                        else:
                            continue

#   Dirichlet/Neumann contribution to RHS
    for field in field_array:

        dirichlet = []
        neumann = []
        LorY = field.LorY

        # Find Dirichlet and Neumann surfaces in region
        # Dirichlet/Neumann surfaces can only be child of region,
        # no point on looking at parent surface
        for s in field.child:
            if surf_array[s].surf_type == 'dirichlet_surface':
                dirichlet.append(s)
            elif surf_array[s].surf_type == 'neumann_surface':
                neumann.append(s)

        if neumann or dirichlet:

            # First look at influence on SIBLING surfaces
            for s in field.child:
                surface = surf_array[s]
                param.kappa = field.kappa

                #nEffect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = -2 * pi * (sd == s)
                    V_diag = 0
                    IorE = 2
                    K_lyr, V_lyr = project(surf_array[sd].phi0,
                                           numpy.zeros(len(surf_array[sd].xi)),
                                           LorY, surf_array[sd], surf_array[s],
                                           K_diag, V_diag, IorE, sd, param,
                                           ind0, timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = locate_s_in_RHS(s, surf_array)
                    s_size = len(surface.xi)

                    # No preconditioner
                    if surface.surf_type in ['dirichlet_surface',
                                             'neumann_surface',
                                             'asc_surface']:
                        F[s_start:s_start + s_size] += K_lyr 
                    else:
                        F[s_start + s_size:s_start + 2 * s_size] += K_lyr 

                
                    # With preconditioner
                    # if s is a charged surface, the surface has only one equation,
                    # else, s has 2 equations and K_lyr affects the external
                    # equation, which is placed after the internal one, hence
                    # Precond[1,:] and Precond[3,:].
                    #if surface.surf_type in ['dirichlet_surface',
                    #                         'neumann_surface',
                    #                         'asc_surface']:
                    #    F[s_start:s_start + s_size] += K_lyr * surface.Precond[0, :]
                    #else:
                    #    F[s_start:s_start + s_size] += K_lyr * surface.Precond[1, :]
                    #    F[s_start + s_size:s_start + 2 * s_size] += K_lyr * surf_array[s].Precond[3, :]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE = 2
                    K_lyr, V_lyr = project(
                        numpy.zeros(len(surf_array[sn].phi0)),
                        surf_array[sn].phi0, LorY, surf_array[sn],
                        surf_array[s], K_diag, V_diag, IorE, sn, param, ind0,
                        timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = locate_s_in_RHS(s, surf_array)
                    s_size = len(surface.xi)

                    # No preconditioner
                    if surface.surf_type in ['dirichlet_surface',
                                             'neumann_surface',
                                             'asc_surface']:
                        F[s_start:s_start + s_size] += -V_lyr 
                    else:
                        F[s_start + s_size:s_start + 2 *
                          s_size] += -V_lyr 

                    # With preconditioner
                    # if s is a charged surface, the surface has only one equation,
                    # else, s has 2 equations and V_lyr affects the external
                    # equation, which is placed after the internal one, hence
                    # Precond[1,:] and Precond[3,:].
                    #if surface.surf_type in ['dirichlet_surface',
                    #                         'neumann_surface',
                    #                         'asc_surface']:
                    #    F[s_start:s_start + s_size] += -V_lyr * surface.Precond[0, :]
                    #else:
                    #    F[s_start:s_start + s_size] += -V_lyr * surface.Precond[1, :]
                    #    F[s_start + s_size:s_start + 2 *
                    #      s_size] += -V_lyr * surface.Precond[3, :]

#           Now look at influence on PARENT surface
#           The dirichlet/neumann surface will never be the parent,
#           since we are not solving for anything inside them.
#           Then, in this case we will not look at self interaction.
            if len(field.parent) == 1:

                s = field.parent[0]

                #               Effect of dirichlet surfaces
                for sd in dirichlet:
                    K_diag = 0
                    V_diag = 0
                    IorE = 1
                    K_lyr, V_lyr = project(surf_array[sd].phi0,
                                           numpy.zeros(len(surf_array[sd].xi)),
                                           LorY, surf_array[sd], surf_array[s],
                                           K_diag, V_diag, IorE, sd, param,
                                           ind0, timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = locate_s_in_RHS(s, surf_array)
                    s_size = len(surf_array[s].xi)

                    # No preconditioner
                    F[s_start:s_start + s_size] += K_lyr 
                    # With preconditioner
                    # Surface s has 2 equations and K_lyr affects the internal
                    # equation, hence Precond[0,:] and Precond[2,:].
                    #F[s_start:s_start + s_size] += K_lyr * surf_array[s].Precond[0, :]
                    #F[s_start + s_size:s_start + 2 *
                    #  s_size] += K_lyr * surf_array[s].Precond[2, :]

#               Effect of neumann surfaces
                for sn in neumann:
                    K_diag = 0
                    V_diag = 0
                    IorE = 1
                    K_lyr, V_lyr = project(
                        numpy.zeros(len(surf_array[sn].phi0)),
                        surf_array[sn].phi0, LorY, surf_array[sn],
                        surf_array[s], K_diag, V_diag, IorE, sn, param, ind0,
                        timing, kernel)

                    # Find location of surface s in RHS array
                    s_start = locate_s_in_RHS(s, surf_array)
                    s_size = len(surf_array[s].xi)

                    # No preconditioner
                    F[s_start:s_start + s_size] += -V_lyr
                    # With preconditioner
                    # Surface s has 2 equations and K_lyr affects the internal
                    # equation, hence Precond[0,:] and Precond[2,:].
                    #F[s_start:s_start + s_size] += -V_lyr * surf_array[s].Precond[0, :]
                    #F[s_start + s_size:s_start + 2 *
                    #  s_size] += -V_lyr * surf_array[s].Precond[2, :]

    return F


def calculate_solvation_energy(surf_array, field_array, param, kernel, output_dir):
    """
    It calculates the solvation energy.

    Arguments
    ----------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    field_array: array, contains the Field classes of each region on the surface.
    param      : class, parameters related to the surface.
    kernel     : pycuda source module.
    output_dir : string, directory where outputs are stored

    Returns
    --------
    E_solv     : float, solvation energy.
    """

    REAL = param.REAL

    par_reac = param
    par_reac.threshold = 0.05
    par_reac.P = 7
    par_reac.theta = 0.0
    par_reac.Nm = (par_reac.P + 1) * (par_reac.P + 2) * (par_reac.P + 3) // 6

    ind_reac = IndexConstant()
    computeIndices(par_reac.P, ind_reac)
    precomputeTerms(par_reac.P, ind_reac)

    par_reac.Nk = 13  # Number of Gauss points per side for semi-analytical integrals

    cal2J = 4.184
    C0 = param.qe**2 * param.Na * 1e-3 * 1e10 / (cal2J * param.E_0)
    E_solv = []

    ff = -1
    for region, f in enumerate(field_array):
        if f.pot == 1:
            parent_type = surf_array[f.parent[0]].surf_type
            if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface':

                E_solv_aux = 0
                ff += 1
                print('Calculating solvation energy for region {}, stored in E_solv[{}]'.format(
                    region, ff))

                AI_int = 0
                Naux = 0
                phi_reac = numpy.zeros(len(f.q))

                #           First look at CHILD surfaces
                #           Need to account for normals pointing outwards
                #           and E_hat coefficient (as region is outside and
                #           dphi_dn is defined inside)
                for i in f.child:
                    s = surf_array[i]
                    s.xk, s.wk = GQ_1D(par_reac.Nk)
                    s.xk = REAL(s.xk)
                    s.wk = REAL(s.wk)
                    for C in range(len(s.tree)):
                        s.tree[C].M = numpy.zeros(par_reac.Nm)
                        s.tree[C].Md = numpy.zeros(par_reac.Nm)

                    Naux += len(s.triangle)

                    #               Coefficient to account for dphi_dn defined in
                    #               interior but calculation done in exterior
                    C1 = s.E_hat

                    if param.GPU == 0:
                        phi_aux, AI = get_phir(s.phi, C1 * s.dphi, s,
                                               f.xq, s.tree, par_reac,
                                               ind_reac)
                    elif param.GPU == 1:
                        phi_aux, AI = get_phir_gpu(s.phi, C1 * s.dphi, s,
                                                   f, par_reac,
                                                   kernel)

                    AI_int += AI
                    phi_reac -= phi_aux  # Minus sign to account for normal pointing out

    #           Now look at PARENT surface
                if len(f.parent) > 0:
                    i = f.parent[0]
                    s = surf_array[i]
                    s.xk, s.wk = GQ_1D(par_reac.Nk)
                    s.xk = REAL(s.xk)
                    s.wk = REAL(s.wk)
                    for C in range(len(s.tree)):
                        s.tree[C].M = numpy.zeros(par_reac.Nm)
                        s.tree[C].Md = numpy.zeros(par_reac.Nm)

                    Naux += len(s.triangle)

                    if param.GPU == 0:
                        phi_aux, AI = get_phir(s.phi, s.dphi, s, f.xq,
                                               s.tree, par_reac, ind_reac)
                    elif param.GPU == 1:
                        phi_aux, AI = get_phir_gpu(
                            s.phi, s.dphi, s, f, par_reac, kernel)

                    AI_int += AI
                    phi_reac += phi_aux

                phi_reac_fname = '{:%Y-%m-%d-%H%M%S}-phi_reac.txt'.format(datetime.now())
                numpy.savetxt(os.path.join(output_dir, phi_reac_fname), phi_reac)
                E_solv_aux += 0.5 * C0 * numpy.sum(f.q * phi_reac)
                E_solv.append(E_solv_aux)

                print('{} of {} analytical integrals for phi_reac calculation in region {}'.format(
                    1. * AI_int / len(f.xq), Naux, region))

    return E_solv


def coulomb_energy(f, param):
    """
    It calculates the Coulomb energy .

    Arguments
    ----------
    f    : class, region in the field array where we desire to calculate the
                  coloumb energy.
    param: class, parameters related to the surface.

    Returns
    --------
    E_coul: float, coloumb energy.
    """

    point_energy = numpy.zeros(len(f.q), param.REAL)
    coulomb_direct(numpy.ravel(f.xq[:, 0]), numpy.ravel(f.xq[:, 1]), numpy.ravel(f.xq[:, 2]), numpy.ravel(f.q), numpy.ravel(point_energy))

    cal2J = 4.184
    C0 = param.qe**2 * param.Na * 1e-3 * 1e10 / (cal2J * param.E_0)

    E_coul = numpy.sum(point_energy) * 0.5 * C0 / (4 * pi * f.E)
    return E_coul


def calculate_surface_energy(surf_array, field_array, param, kernel):
    """
    It calculates the surface energy

    Arguments
    ----------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    field_array: array, contains the Field classes of each region on the surface.
    param      : class, parameters related to the surface.
    kernel     : pycuda source module.

    Returns
    --------
    E_surf: float, surface energy.
    """

    par_reac = Parameters()
    par_reac = param
    par_reac.threshold = 0.05
    par_reac.P = 7
    par_reac.theta = 0.0
    par_reac.Nm = (par_reac.P + 1) * (par_reac.P + 2) * (par_reac.P + 3) / 6

    ind_reac = IndexConstant()
    computeIndices(par_reac.P, ind_reac)
    precomputeTerms(par_reac.P, ind_reac)

    par_reac.Nk = 13  # Number of Gauss points per side for semi-analytical integrals

    cal2J = 4.184
    C0 = param.qe**2 * param.Na * 1e-3 * 1e10 / (cal2J * param.E_0)
    E_surf = []

    ff = -1
    for f in param.E_field:
        parent_surf = surf_array[field_array[f].parent[0]]

        if parent_surf.surf_type in ['dirichlet_surface']:
            ff += 1
            print('Calculating surface energy around region {}, stored in E_surf[{}]'.format(
                f, ff))
            Esurf_aux = -numpy.sum(-parent_surf.Eout * parent_surf.dphi *
                                   parent_surf.phi * parent_surf.area)
            E_surf.append(0.5 * C0 * Esurf_aux)

        if parent_surf.surf_type in ['neumann_surface']:
            ff += 1
            print('Calculating surface energy around region {}, stored in E_surf[{}]'.format(
                f, ff))
            Esurf_aux = numpy.sum(-parent_surf.Eout * parent_surf.dphi *
                                  parent_surf.phi * parent_surf.area)
            E_surf.append(0.5 * C0 * Esurf_aux)
    return E_surf

def dipole_moment(surf_array, electric_field):
    """
    It calculates the dipole moment on a surface and stores it in the 'dipole'
    attribute of the surface class. The dipole is expressed as a boundary
    integral.

    Arguments
    ---------
    surf_array   : array, contains the surface classes of each region on the
                          surface.
    electric_field: float, electric field intensity, it is in the 'z'
                          direction, '-' indicates '-z'.
    """

    for i in range(len(surf_array)):

        s = surf_array[i]
        xc = numpy.array([s.xi, s.yi, s.zi])

        #Changing dphi to outer side of surfaces
        dphi = s.dphi * s.E_hat - (1 - s.E_hat) * electric_field * s.normal[:,2]

        I1 = numpy.sum(xc * dphi * s.area, axis=1)
        I2 = numpy.sum(numpy.transpose(s.normal) * s.phi * s.area, axis=1)

        s.dipole = s.Eout * (I1-I2)


def extinction_cross_section(surf_array, k, n, wavelength, electric_field):
    """
    It computes the extinction cross section (Acording to Mischenko2007).

    Arguments
    ---------
    surf_array    : array, contains the surface classes of each region on the
                           surface.
    k             : array, unit vector in direction of wave propagation.
    n             : array, unit vector in direction of electric field.
    wavelength    : float, wavelength of the incident electric field.
    electric_field: float, electric field intensity, it is in the 'z'
                           direction, '-' indicates '-z'.

    Returns
    -------
    Cext          : list, contains the extinction cross section of surfaces.
    surf_Cext     : list, indices of the surface where Cext is being calculated.
    """

    Cext = []
    surf_Cext = []

    for i in range(len(surf_array)):

        s = surf_array[i]

        diffractionCoeff = numpy.sqrt(s.Eout)
        waveNumber = 2 * numpy.pi * diffractionCoeff / wavelength

        v1 = numpy.cross(k, s.dipole)
        v2 = numpy.cross(v1, k)

        C1 = numpy.dot(n, v2) * waveNumber**2 / (s.Eout * electric_field)

        #multiplying by 0.01 to convert to nm^2
        Cext.append(1 / waveNumber.real * C1.imag * 0.01)
        surf_Cext.append(i)

    return Cext, surf_Cext

def compute_normal_electric_field(surf, s, field, param, sigma, ind0, kernel, timing):

    Nq = len(field.q)

    if param.GPU==1:
        computeRHSKt_gpu = kernel.get_function("compute_RHSKt")
        param.Nround = len(surf.twig)*param.NCRIT
        GSZ = int(numpy.ceil(float(param.Nround)/param.NCRIT)) # CUDA grid size

        Fx_gpu = gpuarray.zeros(param.Nround, dtype=param.REAL)     
        Fy_gpu = gpuarray.zeros(param.Nround, dtype=param.REAL)     
        Fz_gpu = gpuarray.zeros(param.Nround, dtype=param.REAL)     
        computeRHSKt_gpu(Fx_gpu, Fy_gpu, Fz_gpu, field.xq_gpu, field.yq_gpu, field.zq_gpu, field.q_gpu,
                surf.xiDev, surf.yiDev, surf.ziDev, surf.sizeTarDev, numpy.int32(Nq), 
                param.REAL(field.E), numpy.int32(param.NCRIT), numpy.int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 
        aux_x = numpy.zeros(param.Nround)
        aux_y = numpy.zeros(param.Nround)
        aux_z = numpy.zeros(param.Nround)
        Fx_gpu.get(aux_x)
        Fy_gpu.get(aux_y)
        Fz_gpu.get(aux_z)

        ElecField = -aux_x[surf.unsort]*surf.normal[:,0] - \
                    aux_y[surf.unsort]*surf.normal[:,1] - \
                    aux_z[surf.unsort]*surf.normal[:,2]

    elif param.GPU==0:
        ElecField = numpy.zeros(len(surf.triangle))
        for i in range(Nq):
            dx_pq = surf.xi - field.xq[i,0] 
            dy_pq = surf.yi - field.xq[i,1]
            dz_pq = surf.zi - field.xq[i,2]
            R_pq = numpy.sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)

            ElecField += field.q[i]/(R_pq*R_pq*R_pq) * (dx_pq*surf.normal[:,0] \
                                                    + dy_pq*surf.normal[:,1] \
                                                    + dz_pq*surf.normal[:,2])

    if sum(abs(sigma))>1e-10:
        ElecField -= project_Kt(sigma, 1, surf, surf, 
                        0., s, param, ind0, timing, kernel)

    ElecField /= 4*pi

    return ElecField

