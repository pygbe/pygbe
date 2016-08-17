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
        Area_null = s.zero_areas(triangle_raw, Area_null)
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



class Surface():
    """
    Surface class.
    It contains information about the solvent excluded surface.

    Attributes
    -----------

    triangle      : list, indices to triangle vertices.
    vertex        : list, position of vertices.
    XinV          : list, weights input for single layer potential.
    XinK          : list, weights input for double layer potential.
    Xout_int      : list, output vector of interior operators.
    Xout_ext      : list, output vector of exterior operators.
    xi            : list, x component of center.
    yi            : list, y component of center.
    zi            : list, z component of center.
    xj            : list, x component of gauss nodes.
    yj            : list, y component of gauss nodes.
    zj            : list, z component of gauss nodes.
    Area          : list, areas of triangles.
    normal        : list, normal of triangles.
    sglInt_int    : list, singular integrals for V for internal equation.
    sglInt_ext    : list, singular integrals for V for external equation.
    xk            : list, position of gauss points on edges.
    wk            : list, weight of gauss points on edges.
    Xsk           : list, position of gauss points for near singular integrals.
    Wsk           : list, weight of gauss points for near singular integrals.
    tree          : list, tree structure.
    twig          : list, tree twigs.
    xiSort        : list, sorted x component of center.
    yiSort        : list, sorted y component of center.
    ziSort        : list, sorted z component of center.
    xjSort        : list, sorted x component of gauss nodes.
    yjSort        : list, sorted y component of gauss nodes.
    zjSort        : list, sorted z component of gauss nodes.
    xcSort        : list, sorted x component of the box centers according to
                          M2P_list array.
    ycSort        : list, sorted y component of the box centers according to
                          M2P_list array.
    zcSort        : list, sorted z component of the box centers according to
                          M2P_list array.
    AreaSort      : list, sorted array of areas.
    sglInt_intSort: list, sorted array of singular integrals for V for internal
                          equation.
    sglInt_extSort: list, sorted array of singular integrals for V for external
                          equation.
    unsort        : list, array of indices to unsort targets.
    triangleSort  : list, sorted array of triangles.
    sortTarget    : list, indices to sort targets.
    sortSource    : list, indices to sort sources.
    offsetSource  : list, offsets to sorted source array.
    offsetTarget  : list, offsets to sorted target array.
    sizeTarget    : list, number of targets per twig.
    offsetTwigs   : list, offset to twig in P2P list array.
    P2P_list      : list, pointers to twigs for P2P interaction list.
    offsetMlt     : list, offset to multipoles in M2P list array.
    M2P_list      : list, pointers to boxes for M2P interaction list.
    Precond       : list, sparse representation of preconditioner for self
                          interaction block.
    Ein           : float, permitivitty inside surface.
    Eout          : float, permitivitty outside surface.
    E_hat         : float, ratio of Ein/Eout.
    kappa_in      : float, kappa inside surface.
    kappa_out     : float, kappa inside surface.
    LorY_in       : int, Laplace (1) or Yukawa (2) in inner region.
    LorY_out      : int, Laplace (1) or Yukawa (2) in outer region.
    surf_type     : int, Surface type: internal_cavity (=0), stern or
                         dielecric_interface (=1).
    phi0          : list, known surface potential (dirichlet) or derivative of
                          potential (neumann).
    phi           : list, potential on surface.
    dphi          : list, derivative of potential on surface.

    # Device data:

    xiDev        : list, sorted x component of center (on the GPU).
    yiDev        : list, sorted y component of center (on the GPU).
    ziDev        : list, sorted z component of center (on the GPU).
    xjDev        : list, sorted x component of gauss nodes (on the GPU).
    yjDev        : list, sorted y component of gauss nodes (on the GPU).
    zjDev        : list, sorted z component of gauss nodes (on the GPU).
    xcDev        : list, sorted x component of the box centers according to
                         M2P_list array (on the GPU).
    ycDev        : list, sorted y component of the box centers according to
                         M2P_list array (on the GPU).
    zcDev        : list, sorted z component of the box centers according to
                         M2P_list array (on the GPU).
    AreaDev      : list, areas of triangles (on the GPU).
    sglInt_intDev: list, singular integrals for V for internal equation (on the
                         GPU).
    sglInt_extDev: list, singular integrals for V for external equation (on the
                         GPU).
    vertexDev    : list, sorted vertex of the triangles.
    sizeTarDev   : list, number of targets per twig (on the GPU).
    offSrcDev    : list, offsets to sorted source array (on the GPU).
    offMltDev    : list, offset to multipoles in M2P list array (on the GPU).
    offTwgDev    : list, offset to twig in P2P list array (on the GPU).
    M2P_lstDev   : list, pointers to boxes for M2P interaction list (on the GPU).
    P2P_lstDev   : list, pointers to twigs for P2P interaction list (on the GPU).
    xkDev        : list, position of gauss points on edges (on the GPU).
    wkDev        : list, weight of gauss points on edges (on the GPU).
    XskDev       : list, position of gauss points for near singular integrals
                         (on the GPU).
    WskDev       : list, weight of gauss points for near singular integrals (on
                         the GPU).
    kDev         : list, quadrature number of each quadrature point, in order.
                         (on the GPU)
    """

    def __init__(self):
        self.twig = []

    def fill_surface(self, param):
        """
        Fill the surface with all the necessary information to solve it.

        -Set the Gauss points.
        -Generate tree, compute the indices and precompute terms for M2M.
        -Generate preconditioner.
        -Compute the diagonal integral for internal and external equations.

        Arguments
        ----------
        param    : class, parameters related to the surface we are studying.

        """

        self.N = len(self.triangle)
        self.Nj = self.N * param.K
        # Calculate centers
        self.calc_centers()

        self.calc_norms()
        # Set Gauss points (sources)
        self.get_gauss_points(param.K)

        # Calculate distances, get R_C0
        self.calc_distance(param)

        # Generate tree, compute indices and precompute terms for M2M
        self.tree = generateTree(self.xi, self.yi, self.zi, param.NCRIT, param.Nm,
                                 self.N, self.R_C0, self.x_center)
        C = 0
        self.twig = findTwigs(self.tree, C, self.twig, param.NCRIT)

        addSources(self.tree, self.twig, param.K)

        self.xk, self.wk = GQ_1D(param.Nk)
        self.Xsk, self.Wsk = quadratureRule_fine(param.K_fine)

        self.generate_preconditioner()

        tic = time.time()
        sortPoints(self, self.tree, self.twig, param)
        toc = time.time()
        time_sort = toc - tic

        return time_sort

    def calc_centers(self):
        self.xi = numpy.average(self.vertex[self.triangle[:], 0], axis=1)
        self.yi = numpy.average(self.vertex[self.triangle[:], 1], axis=1)
        self.zi = numpy.average(self.vertex[self.triangle[:], 2], axis=1)

    def calc_norms(self):

        L0 = self.vertex[self.triangle[:, 1]] - self.vertex[self.triangle[:, 0]]
        L2 = self.vertex[self.triangle[:, 0]] - self.vertex[self.triangle[:, 2]]

        self.normal = numpy.cross(L0, L2)
        self.Area = numpy.sqrt(numpy.sum(self.normal**2, axis=1)) / 2
        self.normal /= (2 * self.Area[:, numpy.newaxis])

    def calc_distance(self, param):

        self.x_center = numpy.average(numpy.vstack((self.xi,
                                                   self.yi,
                                                   self.zi)), axis=1).astype(param.REAL)
        dist = numpy.sqrt((self.xi - self.x_center[0])**2 +
                          (self.yi - self.x_center[1])**2 +
                          (self.zi - self.x_center[2])**2)
        self.R_C0 = max(dist)

    def get_gauss_points(self, n):
        """
        It gets the Gauss points for far away integrals.

        Arguments
        ----------
        y       : list, vertices of the triangles.
        triangle: list, indices for the corresponding triangles.
        n       : int (1,3,4,7), desired Gauss points per element.

        Returns
        --------
        xi[:,0] : position of the gauss point in the x axis.
        xi[:,1] : position of the gauss point in the y axis.
        xi[:,2] : position of the gauss point in the z axis.
        """

        #N  = len(triangle) # Number of triangles
        gauss_array = numpy.zeros((self.N*n,3))
        if n==1:
            gauss_array = numpy.average(self.vertex[self.triangle], axis=1)

        elif n==3:
            for i in range(self.N):
                M = self.vertex[self.triangle[i]]
                gauss_array[n*i, :] = numpy.dot(M.T, numpy.array([0.5, 0.5, 0.]))
                gauss_array[n*i+1, :] = numpy.dot(M.T, numpy.array([0., 0.5, 0.5]))
                gauss_array[n*i+2, :] = numpy.dot(M.T, numpy.array([0.5, 0., 0.5]))

        elif n==4:
            for i in range(self.N):
                M = self.vertex[self.triangle[i]]
                gauss_array[n*i, :] = numpy.dot(M.T, numpy.array([1/3., 1/3., 1/3.]))
                gauss_array[n*i+1, :] = numpy.dot(M.T, numpy.array([3/5., 1/5., 1/5.]))
                gauss_array[n*i+2, :] = numpy.dot(M.T, numpy.array([1/5., 3/5., 1/5.]))
                gauss_array[n*i+3, :] = numpy.dot(M.T, numpy.array([1/5., 1/5., 3/5.]))

        elif n==7:
            for i in range(self.N):
                M = self.vertex[self.triangle[i]]
                gauss_array[n*i+0, :] = numpy.dot(M.T, numpy.array([1/3.,1/3.,1/3.]))
                gauss_array[n*i+1, :] = numpy.dot(M.T, numpy.array([.797426985353087, .101286507323456, .101286507323456]))
                gauss_array[n*i+2, :] = numpy.dot(M.T, numpy.array([.101286507323456, .797426985353087, .101286507323456]))
                gauss_array[n*i+3, :] = numpy.dot(M.T, numpy.array([.101286507323456, .101286507323456, .797426985353087]))
                gauss_array[n*i+4, :] = numpy.dot(M.T, numpy.array([.059715871789770, .470142064105115, .470142064105115]))
                gauss_array[n*i+5, :] = numpy.dot(M.T, numpy.array([.470142064105115, .059715871789770, .470142064105115]))
                gauss_array[n*i+6, :] = numpy.dot(M.T, numpy.array([.470142064105115, .470142064105115, .059715871789770]))

        self.xj, self.yj, self.zj = gauss_array.T


    def generate_preconditioner(self):
        # Generate preconditioner
        # Will use block-diagonal preconditioner (AltmanBardhanWhiteTidor2008)
        # If we have complex dielectric constants we need to initialize Precon with
        # complex type else it'll be float.
        if type(self.E_hat) == complex:
            self.Precond = numpy.zeros((4, self.N), complex)
        else:
            self.Precond = numpy.zeros((4, self.N))
        # Stores the inverse of the block diagonal (also a tridiag matrix)
        # Order: Top left, top right, bott left, bott right
        centers = numpy.vstack((self.xi, self.yi, self.zi)).T

        #   Compute diagonal integral for internal equation
        VL = numpy.zeros(self.N)
        KL = numpy.zeros(self.N)
        VY = numpy.zeros(self.N)
        KY = numpy.zeros(self.N)
        computeDiagonal(VL, KL, VY, KY, numpy.ravel(self.vertex[self.triangle[:]]),
                        numpy.ravel(centers), self.kappa_in, 2 * numpy.pi, 0.,
                        self.xk, self.wk)
        if self.LorY_in == 1:
            dX11 = KL
            dX12 = -VL
            self.sglInt_int = VL  # Array for singular integral of V through interior
        elif self.LorY_in == 2:
            dX11 = KY
            dX12 = -VY
            self.sglInt_int = VY  # Array for singular integral of V through interior
        else:
            self.sglInt_int = numpy.zeros(self.N)

    #   Compute diagonal integral for external equation
        VL = numpy.zeros(self.N)
        KL = numpy.zeros(self.N)
        VY = numpy.zeros(self.N)
        KY = numpy.zeros(self.N)
        computeDiagonal(VL, KL, VY, KY, numpy.ravel(self.vertex[self.triangle[:]]),
                        numpy.ravel(centers), self.kappa_out, 2 * numpy.pi, 0.,
                        self.xk, self.wk)
        if self.LorY_out == 1:
            dX21 = KL
            dX22 = self.E_hat * VL
            self.sglInt_ext = VL  # Array for singular integral of V through exterior
        elif self.LorY_out == 2:
            dX21 = KY
            dX22 = self.E_hat * VY
            self.sglInt_ext = VY  # Array for singular integral of V through exterior
        else:
            self.sglInt_ext = numpy.zeros(N)

        if self.surf_type != 'dirichlet_surface' and self.surf_type != 'neumann_surface':
            d_aux = 1 / (dX22 - dX21 * dX12 / dX11)
            self.Precond[0, :] = 1 / dX11 + 1 / dX11 * dX12 * d_aux * dX21 / dX11
            self.Precond[1, :] = -1 / dX11 * dX12 * d_aux
            self.Precond[2, :] = -d_aux * dX21 / dX11
            self.Precond[3, :] = d_aux
        elif self.surf_type == 'dirichlet_surface':
            self.Precond[0, :] = 1 / VY  # So far only for Yukawa outside
        elif self.surf_type == 'neumann_surface' or self.surf_type == 'asc_surface':
            self.Precond[0, :] = 1 / (2 * numpy.pi)

    def zero_areas(self, triangle_raw, area_null):
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
            L0 = self.vertex[triangle_raw[i, 1]] - self.vertex[triangle_raw[i, 0]]
            L2 = self.vertex[triangle_raw[i, 0]] - self.vertex[triangle_raw[i, 2]]
            normal_aux = numpy.cross(L0, L2)
            area_aux = linalg.norm(normal_aux) / 2
            if area_aux < 1e-10:
                area_null.append(i)

        return area_null
