"""
It contains the functions to build the tree and compute all the interactions.
"""
import time
import numpy
from scipy.special import comb

# Wrapped code
from pygbe.tree.multipole import multipole_c, setIndex, getIndex_arr, multipole_sort, multipoleKt_sort
from pygbe.tree.direct import direct_c, direct_sort, directKt_sort
from pygbe.tree.calculateMultipoles import P2M, M2M

# CUDA libraries
try:
    import pycuda.driver as cuda
except:
    print('PyCUDA not installed, performance might not be so hot')


class Cell():
    """
    Cell class. It contains the information about the cells in the tree.

    Attributes
    -----------
    nsource   : int, number of source particles.
    ntarget   : int, number of target particles.
    nchild    : int, number of child boxes in binary, 8bit value, if certain
                     child exists, that bit will be 1.
    source    : array, pointer to source particles.
    target    : array, pointer to target particles.
    xc        : float, x position of the center of cell.
    yc        : float, y position of the center of cell.
    zc        : float, z position of the center of cell.
    r         : float, cell radius, i.e half length.
    parent    : int, pointer to parent cell.
    child     : array, pointer to child cell.
    M         : array, array with multipoles.
    Md        : array, array with multipoles for grad(G).n.
    P2P_list  : list, pointer to cells that interact with P2P.
    M2P_list  : list, pointer to cells that interact with M2P.
    M2P_size  : list, size of the M2P interaction list.
    list_ready: int, flag to know if P2P list is already generated.
    twig_array: list, position in the twig array.
    """

    def __init__(self, NCRIT, Nm):
        """
        NCRIT: int, maximum number of boundary elements per twig box of tree
                    structure.
        Nm   : int, number of multipole coefficients.
        """
        self.nsource = 0  # Number of source particles
        self.ntarget = 0  # Number of target particles
        self.nchild = 0  # Number of child boxes in binary
        # This will be a 8bit value and if certain
        # child exists, that bit will be 1.

        self.source = numpy.array([], dtype=numpy.int32
                                  )  # Pointer to source particles
        self.target = numpy.array([], dtype=numpy.int32
                                  )  # Pointer to target particles
        self.xc = 0.  # x position of cell
        self.yc = 0.  # y position of cell
        self.zc = 0.  # z position of cell
        self.r = 0.  # cell radius

        self.parent = 0  # Pointer to parent cell
        self.child = numpy.zeros(8, dtype=numpy.int32)  # Pointer to child cell

        self.M = numpy.zeros(Nm)  # Array with multipoles
        self.Md = numpy.zeros(Nm)  # Array with multipoles for grad(G).n

        self.P2P_list = numpy.array(
            [], dtype=numpy.int32)  # Pointer to cells that interact with P2P
        self.M2P_list = []  # Pointer to cells that interact with M2P
        self.M2P_size = []  # Size of the M2P interaction list
        self.list_ready = 0  # Flag to know if P2P list is already generated
        self.twig_array = []  # Position in the twig array


def add_child(octant, Cells, i, NCRIT, Nm, Ncell):
    """
    It adds a child cell to the Cells.

    Arguments
    ----------
    octant: int, octant of the child cell.
    Cells : array, it contains the cells information.
    i     : int, index of parent cell in Cells array.
    NCRIT : int, maximum number of boundary elements per twig box of tree
                 structure.
    Nm    : int, number of multipole coefficients.
    Ncell : int, number of cells in the tree.

    Returns
    --------
    Ncell : int, number of cells in the tree.
    """

    CN = Cell(NCRIT, Nm)  # CN: child cell
    CN.r = Cells[i].r / 2
    CN.xc = Cells[i].xc + CN.r * (
        (octant & 1) * 2 - 1)  # octant&X returns X if true
    CN.yc = Cells[i].yc + CN.r * (
        (octant & 2) - 1)  # Want to make ((octant&X)*Y - Z)=1
    CN.zc = Cells[i].zc + CN.r * ((octant & 4) / 2 - 1)
    CN.parent = i
    Cells[i].child[octant] = Ncell
    Cells[i].nchild |= (1 << octant)
    Cells[Ncell] = CN
    Ncell += 1

    return Ncell


def split_cell(x, y, z, Cells, C, NCRIT, Nm, Ncell):
    """
    It splits a cell with more than (>=) NCRIT particles.
    Particles in this context means boundary elements.

    Arguments
    ----------
    x    : array, x position of the particles.
    y    : array, y position of the particles.
    z    : array, z position of the particles.
    Cells: array, it contains the cells information.
    C    : int, index in the Cells array of the cell to be splitted .
    NCRIT: int, maximum number of boundary elements per twig box of tree
                structure.
    Nm   : int, number of multipole coefficients.
    Ncell: int, number of cells in the tree.

    Returns
    --------
    Ncell: int, number of cells in the tree.
    """

    for l in Cells[C].target:
        octant = int(x[l] > Cells[C].xc) + int(y[l] > Cells[C].yc) * 2 + int(z[
            l] > Cells[C].zc) * 4
        if (not (Cells[C].nchild &
                 (1 << octant))):  # Ask if octant exists already
            Ncell = add_child(octant, Cells, C, NCRIT, Nm, Ncell)

        CC = Cells[C].child[octant]  # Pointer to child cell
        Cells[CC].target = numpy.append(Cells[CC].target, l)
        Cells[CC].ntarget += 1

        if (Cells[CC].ntarget >= NCRIT):
            Ncell = split_cell(x, y, z, Cells, CC, NCRIT, Nm, Ncell)

    return Ncell


def generateTree(xi, yi, zi, NCRIT, Nm, N, radius, x_center):
    """
    It generates a target-based tree.

    Arguments
    ----------
    xi      : array, x position of the targets, i.e collocation points.
    yi      : array, y position of the targets, i.e collocation points.
    zi      : array, z position of the targets, i.e collocation points.
    NCRIT   : int, maximum number of boundary elements per twig box of tree
                   structure.
    Nm      : int, number of multipole coefficients.
    r       : float, cell radius, i.e half length.
    x_center: array, center of the root cell.

    Returns
    --------
    Cells   : array, cells of the tree.
    """

    C0 = Cell(NCRIT, Nm)
    C0.xc = x_center[0]
    C0.yc = x_center[1]
    C0.zc = x_center[2]
    C0.r = radius

    Cells = [Cell(NCRIT, Nm)] * N
    Cells[0] = C0
    Ncell = 1
    for i in range(N):

        C = 0
        while (Cells[C].ntarget >= NCRIT):
            Cells[C].ntarget += 1

            octant = int(xi[i] > Cells[C].xc) + int(yi[i] > Cells[
                C].yc) * 2 + int(zi[i] > Cells[C].zc) * 4
            if (not (Cells[C].nchild & (1 << octant))):
                Ncell = add_child(octant, Cells, C, NCRIT, Nm, Ncell)

            C = Cells[C].child[octant]

        Cells[C].target = numpy.append(Cells[C].target, i)
        Cells[C].ntarget += 1

        if (Cells[C].ntarget >= NCRIT):
            Ncell = split_cell(xi, yi, zi, Cells, C, NCRIT, Nm, Ncell)

    Cells = Cells[:Ncell]
    return Cells


def findTwigs(Cells, C, twig, NCRIT):
    """
    It finds the twig cells, the ones that have < NCRIT.

    Arguments
    ----------
    Cells: array, cells of the tree.
    C    : int, index of cell in the Cells array.
    twig : array, indices of twigs in Cells array.
    NCRIT: int, maximum number of boundary elements per twig box of tree
                structure.

    Returns
    --------
    twig : array, indices of twigs in Cells array.
    """

    if (Cells[C].ntarget >= NCRIT):
        for c in range(8):
            if (Cells[C].nchild & (1 << c)):
                twig = findTwigs(Cells, Cells[C].child[c], twig, NCRIT)
    else:
        twig.append(C)
        Cells[C].twig_array = numpy.int32(len(twig) - 1)

    return twig


def addSources(Cells, twig, K):
    """
    It adds the source points to the cells.
    Puts the sources in the same cell as the collocation point of the same
    panel.
    This version works fast when used with sorted arrays.

    Arguments
    ----------
    Cells: array, cells of the tree.
    twig : array, indices of twigs in Cells array.
    K    : int, number of Gauss points per element.
    """

    for C in twig:
        Cells[C].nsource = K * Cells[C].ntarget
        for j in range(K):
            Cells[C].source = numpy.append(Cells[C].source,
                                           K * Cells[C].target + j)


def addSources2(x, y, z, j, Cells, C, NCRIT):
    """
    It adds the source points to the cells.
    Puts the sources in the same cell as the collocation point of the same
    panel.
    This version is a generic version that loop over the cells looking for
    twigs and sets the sources one it finds a twig.

    Arguments
    ----------
    x    : array, x coordinate of the sources.
    y    : array, y coordinate of the sources.
    z    : array, z coordinate of the sources.
    j    : int, index of the source in the source array.
    Cells: array, cells of the tree.
    C    : int, index of cell in the Cells array.
    NCRIT: int, maximum number of boundary elements per twig box of tree
                structure.
    """

    if (Cells[C].ntarget >= NCRIT):
        octant = (x[j] > Cells[C].xc) + ((y[j] > Cells[C].yc) << 1) + (
            (z[j] > Cells[C].zc) << 2)
        if (Cells[C].nchild & (1 << octant)):  # If child cell exists, use
            O = octant
        else:  # If child cell doesn't exist add to closest existing child
            r = []
            child = []
            for c in range(8):
                if (Cells[C].nchild & (1 << c)):
                    dx = x[j] - Cells[Cells[C].child[c]].xc
                    dy = y[j] - Cells[Cells[C].child[c]].yc
                    dz = z[j] - Cells[Cells[C].child[c]].zc
                    r.append(numpy.sqrt(dx * dx + dy * dy + dz * dz))
                    child.append(c)
            close_child = r.index(min(r))  # Find index of closest child
            O = child[close_child]
        addSources2(x, y, z, j, Cells, Cells[C].child[O], NCRIT)

    else:
        Cells[C].nsource += 1
        Cells[C].source = numpy.append(Cells[C].source, j)


def addSources3(x, y, z, Cells, twig):
    """
    It adds the source points to the cells.
    Puts the sources in the same cell as the collocation point of the same
    panel.
    This version works fast when we uses the twig_array, array that contains
    the location in the cells array of the twig cells.

    Arguments
    ----------
    x    : array, x coordinate of the sources.
    y    : array, y coordinate of the sources.
    z    : array, z coordinate of the sources.
    j    : int, index of the source in the source array.
    Cells: array, cells of the tree.
    twig : array, indices of twigs in Cells array.
    """

    dx = numpy.zeros((len(twig), len(x)))
    dy = numpy.zeros((len(twig), len(x)))
    dz = numpy.zeros((len(twig), len(x)))
    j = 0

    for t in twig:
        dx[j] = x - Cells[t].xc
        dy[j] = y - Cells[t].yc
        dz[j] = z - Cells[t].zc
        j += 1
    r = numpy.sqrt(dx * dx + dy * dy + dz * dz)

    close_twig = numpy.argmin(r, axis=0)

    for j in range(len(close_twig)):
        Cells[twig[close_twig[j]]].nsource += 1
        Cells[twig[close_twig[j]]].source = numpy.append(
            Cells[twig[close_twig[j]]].source, j)


def sortPoints(surface, Cells, twig, param):
    """
    It sort the target and source points.

    We sort them to makes the computation easy and faster in the GPU.
    We unsort them when we need the information to be analized after it was
    computed.

    Arguments
    ----------
    surface: class, surface that we are analysing.
    Cells  : array, cells of the tree.
    twig   : array, indices of twigs in Cells array.
    param  : class, parameters related to the surface.
    """

    Nround = len(twig) * param.NCRIT

    surface.sortTarget = numpy.zeros(Nround, dtype=numpy.int32)
    surface.unsort = numpy.zeros(len(surface.xi), dtype=numpy.int32)
    surface.sortSource = numpy.zeros(len(surface.xj), dtype=numpy.int32)
    surface.offsetSource = numpy.zeros(len(twig) + 1, dtype=numpy.int32)
    surface.offsetTarget = numpy.zeros(len(twig), dtype=numpy.int32)
    surface.sizeTarget = numpy.zeros(len(twig), dtype=numpy.int32)
    offSrc = 0
    i = 0
    for C in twig:
        surface.sortTarget[param.NCRIT * i:param.NCRIT * i + Cells[
            C].ntarget] = Cells[C].target
        surface.unsort[Cells[C].target] = range(
            param.NCRIT * i, param.NCRIT * i + Cells[C].ntarget)
        surface.sortSource[offSrc:offSrc + Cells[C].nsource] = Cells[C].source
        offSrc += Cells[C].nsource
        surface.offsetSource[i + 1] = offSrc
        surface.offsetTarget[i] = i * param.NCRIT
        surface.sizeTarget[i] = Cells[C].ntarget
        i += 1

    surface.xiSort = surface.xi[surface.sortTarget]
    surface.yiSort = surface.yi[surface.sortTarget]
    surface.ziSort = surface.zi[surface.sortTarget]
    surface.xjSort = surface.xj[surface.sortSource]
    surface.yjSort = surface.yj[surface.sortSource]
    surface.zjSort = surface.zj[surface.sortSource]
    surface.AreaSort = surface.area[surface.sortSource // param.K]
    surface.sglInt_intSort = surface.sglInt_int[surface.sortSource // param.K]
    surface.sglInt_extSort = surface.sglInt_ext[surface.sortSource // param.K]
    surface.triangleSort = surface.triangle[surface.sortSource // param.K]


def computeIndices(P, ind0):
    """
    It computes the indices (exponents) needed to compute the Taylor expansion.

    Arguments
    ----------
    P   : int, order of the Taylor expansion.
    ind0: class, it contains the indices related to the treecode computation.
    """

    II = []
    JJ = []
    KK = []
    index = []
    ind0.index_large = numpy.zeros(
        (P + 1) * (P + 1) *
        (P + 1), dtype=numpy.int32)
    for ii in range(P + 1):
        for jj in range(P + 1 - ii):
            for kk in range(P + 1 - ii - jj):
                index.append(setIndex(P, ii, jj, kk))
                II.append(ii)
                JJ.append(jj)
                KK.append(kk)
                ind0.index_large[(P + 1) * (P + 1) * ii + (P + 1) * jj +
                                 kk] = index[-1]

    ind0.II = numpy.array(II, numpy.int32)
    ind0.JJ = numpy.array(JJ, numpy.int32)
    ind0.KK = numpy.array(KK, numpy.int32)
    ind0.index = numpy.array(index, numpy.int32)
#    index = getIndex_arr(P,II,JJ,KK)


def precomputeTerms(P, ind0):
    """
    It precomputes the terms for P2M and M2M computation.

    Arguments
    ----------
    P   : int, order of the Taylor expansion.
    ind0: class, it contains the indices related to the treecode computation.
    """

    # Precompute terms for
    ind0.combII = numpy.array([], dtype=numpy.int32)
    ind0.combJJ = numpy.array([], dtype=numpy.int32)
    ind0.combKK = numpy.array([], dtype=numpy.int32)
    ind0.IImii = numpy.array([], dtype=numpy.int32)
    ind0.JJmjj = numpy.array([], dtype=numpy.int32)
    ind0.KKmkk = numpy.array([], dtype=numpy.int32)
    ind0.index_small = numpy.array([], dtype=numpy.int32)
    ind0.index_ptr = numpy.zeros(len(ind0.II) + 1, dtype=numpy.int32)
    for i in range(len(ind0.II)):
        ii, jj, kk = numpy.mgrid[0:ind0.II[i] + 1:1, 0:ind0.JJ[i] + 1:1, 0:
                                 ind0.KK[i] + 1:1].astype(numpy.int32)
        ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
        index_aux = numpy.zeros(len(ii), numpy.int32)
        getIndex_arr(P, len(ii), index_aux, ii, jj, kk)
        ind0.index_small = numpy.append(ind0.index_small, index_aux)
        ind0.index_ptr[i + 1] = len(index_aux) + ind0.index_ptr[i]
        ind0.combII = numpy.append(ind0.combII, comb(ind0.II[i], ii))
        ind0.combJJ = numpy.append(ind0.combJJ, comb(ind0.JJ[i], jj))
        ind0.combKK = numpy.append(ind0.combKK, comb(ind0.KK[i], kk))
        ind0.IImii = numpy.append(ind0.IImii, ind0.II[i] - ii)
        ind0.JJmjj = numpy.append(ind0.JJmjj, ind0.JJ[i] - jj)
        ind0.KKmkk = numpy.append(ind0.KKmkk, ind0.KK[i] - kk)


def interactionList(surfSrc, surfTar, CJ, CI, theta, NCRIT, offTwg, offMlt,
                    s_src):
    """
    It finds the list of cells which each twig cell interacts.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    CJ     : int, index of source cell.
    CI     : int, index of target cell.
    theta  : float, Multipole-acceptance criterion (MAC).
    NCRIT  : int, maximum number of boundary elements per twig box of tree
                  structure.
    offTwg : array, pointer to the first element in the array P2P_list which
                    contains the P2P interaction list for each twig cell.
    offMlt : array, pointer to the first element in the array M2P_list which
                    contains the M2P interaction list for each twig cell.
    s_src  : int, position (index) in the surface-array of the surface that
                  contains the sources.

    Returns
    --------
    offTwg : array, pointer to the first element in the P2P interaction list
                    for each twig cell.
    offMlt : array, pointer to the first element in the M2P interaction list
                    for each twig cell.
    """

    if (surfSrc.tree[CJ].ntarget >= NCRIT):
        for c in range(8):
            if (surfSrc.tree[CJ].nchild & (1 << c)):
                CC = surfSrc.tree[CJ].child[c]  # Points at child cell
                dxi = surfSrc.tree[CC].xc - surfTar.tree[CI].xc
                dyi = surfSrc.tree[CC].yc - surfTar.tree[CI].yc
                dzi = surfSrc.tree[CC].zc - surfTar.tree[CI].zc
                r = numpy.sqrt(dxi * dxi + dyi * dyi + dzi * dzi)
                if surfTar.tree[CI].r + surfSrc.tree[
                        CC].r > theta * r:  # Max distance between particles
                    offTwg, offMlt = interactionList(
                        surfSrc, surfTar, CC, CI, theta, NCRIT, offTwg, offMlt,
                        s_src)
                else:
                    surfTar.M2P_list[s_src, offMlt] = CC
                    offMlt += 1
    else:
        twig_cell = surfSrc.tree[CJ].twig_array
        surfTar.P2P_list[s_src, offTwg] = twig_cell
        offTwg += 1

    return offTwg, offMlt


def generateList(surf_array, field_array, param):
    """
    Loops over the surfaces to then compute the interactionList().

    Arguments
    ----------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    field_array: array, contains the Field classes of each region on the
                 surface.
    param      : class, parameters related to the surface.
    """

    Nsurf = len(surf_array)
    Nfield = len(field_array)

    # Allocate data
    maxTwigSize = 0
    for i in range(Nsurf):
        maxTwigSize = max(len(surf_array[i].twig), maxTwigSize)
        maxTwigSize = max(len(surf_array[i].tree), maxTwigSize)

    for i in range(Nsurf):
        surf_array[i].P2P_list = numpy.zeros(
            (Nsurf, maxTwigSize * maxTwigSize),
            dtype=numpy.int32)
        surf_array[i].offsetTwigs = numpy.zeros(
            (Nsurf,
             maxTwigSize + 1), dtype=numpy.int32)
        surf_array[i].M2P_list = numpy.zeros(
            (Nsurf, maxTwigSize * maxTwigSize),
            dtype=numpy.int32)
        surf_array[i].offsetMlt = numpy.zeros(
            (Nsurf,
             maxTwigSize + 1), dtype=numpy.int32)
        for CI in surf_array[i].twig:
            surf_array[i].tree[CI].M2P_list = numpy.zeros(
                (Nsurf, maxTwigSize), dtype=numpy.int32)
            surf_array[i].tree[CI].M2P_size = numpy.zeros(Nsurf,
                                                          dtype=numpy.int32)

    # Generate list
    # Non-self interaction
    for i in range(Nfield):
        S = []
        S[:] = field_array[i].child[:]  # Children surfaces
        if len(field_array[i].parent) > 0:
            S.append(field_array[i].parent[0])  # Parent surface

        for s_tar in S:  # Loop over surfaces
            for s_src in S:
                offTwg = 0
                offMlt = 0
                ii = 0
                for CI in surf_array[s_tar].twig:
                    if s_src != s_tar:  # Non-self interaction
                        CJ = 0
                        offTwg, offMlt = interactionList(
                            surf_array[s_src], surf_array[s_tar], CJ, CI,
                            param.theta, param.NCRIT, offTwg, offMlt, s_src)
                        surf_array[s_tar].offsetTwigs[s_src, ii + 1] = offTwg
                        surf_array[s_tar].offsetMlt[s_src, ii + 1] = offMlt
                        ii += 1

    # Self interaction
    for s in range(Nsurf):
        offTwg = 0
        offMlt = 0
        ii = 0
        for CI in surf_array[s].twig:
            CJ = 0
            offTwg, offMlt = interactionList(surf_array[s], surf_array[s], CJ,
                                             CI, param.theta, param.NCRIT,
                                             offTwg, offMlt, s)
            surf_array[s].offsetTwigs[s, ii + 1] = offTwg
            surf_array[s].offsetMlt[s, ii + 1] = offMlt
            ii += 1

    for s_tar in range(Nsurf):
        surf_array[s_tar].xcSort = numpy.zeros((Nsurf, maxTwigSize *
                                                maxTwigSize))
        surf_array[s_tar].ycSort = numpy.zeros((Nsurf, maxTwigSize *
                                                maxTwigSize))
        surf_array[s_tar].zcSort = numpy.zeros((Nsurf, maxTwigSize *
                                                maxTwigSize))
        for s_src in range(Nsurf):
            M2P_size = surf_array[s_tar].offsetMlt[s_src, len(surf_array[
                s_tar].twig)]
            i = -1
            for C in surf_array[s_tar].M2P_list[s_src, 0:M2P_size]:
                i += 1
                surf_array[s_tar].xcSort[s_src, i] = surf_array[s_src].tree[
                    C].xc
                surf_array[s_tar].ycSort[s_src, i] = surf_array[s_src].tree[
                    C].yc
                surf_array[s_tar].zcSort[s_src, i] = surf_array[s_src].tree[
                    C].zc


def getMultipole(Cells, C, x, y, z, mV, mKx, mKy, mKz, ind0, P, NCRIT):
    """
    It gets the multipole of the twig cells.
    (P2M)

    Note: In this context when we refer to mass we mean
                 mass  = (vector x gauss weights)

          where 'vector' is the vector in the matrix-vector multiplication in
          the GMRES.

    Arguments
    ----------
    Cells: array, it contains the cells information.
    C    : int, index of the cell in the Cells array.
    x    : array, x coordinate of the sources.
    y    : array, y coordinate of the sources.
    z    : array, z coordinate of the sources.
    mV   : array, mass of the source particle for the single layer potential
                  calculation.
    mKx  : array, mass of the source particle times  the 'x' component of the
                  normal vector, for the double layer potential calculation.
    mKy  :array, mass of the source particle times  the 'y' component of the
                  normal vector, for the double layer potential calculation.
    mKz  :array, mass of the source particle times  the 'z' component of the
                  normal vector, for the double layer potential calculation.
    ind0 : class, it contains the indices related to the treecode computation.
    P    : int, order of the Taylor expansion.
    NCRIT: int, maximum number of boundary elements per twig box of tree
                structure.
    """

    if (Cells[C].ntarget >= NCRIT):

        Cells[C].M[:] = 0.0  # Initialize multipoles
        Cells[C].Md[:] = 0.0

        for c in range(8):
            if (Cells[C].nchild & (1 << c)):
                getMultipole(Cells, Cells[C].child[c], x, y, z, mV, mKx, mKy,
                             mKz, ind0, P, NCRIT)
    else:

        Cells[C].M[:] = 0.0  # Initialize multipoles
        Cells[C].Md[:] = 0.0

        l = Cells[C].source
        P2M(Cells[C].M, Cells[C].Md, x[l], y[l], z[l], mV[l], mKx[l], mKy[l],
            mKz[l], Cells[C].xc, Cells[C].yc, Cells[C].zc, ind0.II, ind0.JJ,
            ind0.KK)


def upwardSweep(Cells, CC, PC, P, II, JJ, KK, index, combII, combJJ, combKK,
                IImii, JJmjj, KKmkk, index_small, index_ptr):
    """
    It calculates the M2M. Calculation of the multipole for non-twig cells .

    Arguments
    ----------
    Cells      : array, it contains the cells information.
    CC         : int, index of the child cell in the Cells array .
    PC         : int, index of the parent cell in the Cells array
    P          : int, order of the Taylor expansion.
    II         : list, multipole order in the x-direction for the treecode.
    JJ         : list, multipole order in the y-direction for the treecode.
    KK         : list, multipole order in the z-direction for the treecode.
    index      : list, pointers to the location of the mulipole of order i,j,k
                       in the multipole array.
    combII     : array, combinatory of (I, i) where I is the maximum i multipole.
    combJJ     : array, combinatory of (J, j) where J is the maximum j multipole.
    combKK     : array, combinatory of (K, k) where K is the maximum k multipole.
    IImii      : array, I-i where I is the maximum i multipole.
    JJmjj      : array, J-j where J is the maximum j multipole.
    KKmkk      : array, K-k where K is the maximum k multipole.
    index_small: list, pointers to the position of multipole order i, j, k
                       in the multipole array, organized in a 1D array which is
                       compressed with respect to index_large (does not consider
                       combinations of i,j,k which do not have a multipole).
    index_ptr  : list, pointer to index_small. Data in index_small is organized
                      in a i-major fashion (i,j,k), and index_ptr points at the
                      position in index_small where the order i changes.
    """

    dx = Cells[PC].xc - Cells[CC].xc
    dy = Cells[PC].yc - Cells[CC].yc
    dz = Cells[PC].zc - Cells[CC].zc

    M2M(Cells[PC].M, Cells[CC].M, dx, dy, dz, II, JJ, KK, combII, combJJ,
        combKK, IImii, JJmjj, KKmkk, index_small, index_ptr)
    M2M(Cells[PC].Md, Cells[CC].Md, dx, dy, dz, II, JJ, KK, combII, combJJ,
        combKK, IImii, JJmjj, KKmkk, index_small, index_ptr)


def M2P_sort(surfSrc, surfTar, K_aux, V_aux, surf, index, param, LorY, timing):
    """
    It computes the far field contribution of the double and single layer
    potential using the sorted data.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    K_aux  : array, far field contribution to the double layer potential.
    V_aux  : array, far field contribution to the single layer potential.
    surf   : int, position of the source surface in the surface array.
    index  : list, pointers to the location of the mulipole of order i,j,k
                   in the multipole array.
    param  : class, parameters related to the surface.
    LorY   : int, Laplace (1) or Yukawa (2).
    timing : class, it contains timing information for different parts of
                    the code.

    Returns
    --------
    K_aux  : array, far field contribution to the double layer potential.
    V_aux  : array, far field contribution to the single layer potential.
    """

    tic = time.time()
    M2P_size = surfTar.offsetMlt[surf, len(surfTar.twig)]
    MSort = numpy.zeros(param.Nm * M2P_size)
    MdSort = numpy.zeros(param.Nm * M2P_size)

    i = -1
    for C in surfTar.M2P_list[surf, 0:M2P_size]:
        i += 1
        MSort[i * param.Nm:i * param.Nm + param.Nm] = surfSrc.tree[C].M
        MdSort[i * param.Nm:i * param.Nm + param.Nm] = surfSrc.tree[C].Md

    multipole_sort(K_aux, V_aux, surfTar.offsetTarget, surfTar.sizeTarget,
                   surfTar.offsetMlt[surf], MSort, MdSort, surfTar.xiSort,
                   surfTar.yiSort, surfTar.ziSort, surfTar.xcSort[surf],
                   surfTar.ycSort[surf], surfTar.zcSort[surf], index, param.P,
                   param.kappa, int(param.Nm), int(LorY))

    toc = time.time()
    timing.time_M2P += toc - tic

    return K_aux, V_aux


def M2PKt_sort(surfSrc, surfTar, Ktx_aux, Kty_aux, Ktz_aux, surf, index, param,
               LorY, timing):
    """
    It computes the far field contribution of the adjoint double potential
    using the sorted data.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    Ktx_aux: array, x component of the far field contribution to the adjoint
                    double layer potential.
    Kty_aux: array, y component of the far field contribution to the adjoint
                    double layer potential.
    Ktz_aux: array, z component of the far field contribution to the adjoint
                    double layer potential.
    surf   : int, position of the source surface in the surface array.
    index  : list, pointers to the location of the mulipole of order i,j,k
                   in the multipole array.
    param  : class, parameters related to the surface.
    LorY   : int, Laplace (1) or Yukawa (2).
    timing : class, it contains timing information for different parts of
                    the code.

    Returns
    --------
    Ktx_aux: array, x component of the far field contribution to the adjoint
                    double layer potential.
    Kty_aux: array, y component of the far field contribution to the adjoint
                    double layer potential.
    Ktz_aux: array, z component of the far field contribution to the adjoint
                    double layer potential.
    """

    tic = time.time()
    M2P_size = surfTar.offsetMlt[surf, len(surfTar.twig)]
    MSort = numpy.zeros(param.Nm * M2P_size)

    i = -1
    for C in surfTar.M2P_list[surf, 0:M2P_size]:
        i += 1
        MSort[i * param.Nm:i * param.Nm + param.Nm] = surfSrc.tree[C].M

    multipoleKt_sort(Ktx_aux, Kty_aux, Ktz_aux, surfTar.offsetTarget,
                     surfTar.sizeTarget, surfTar.offsetMlt[surf], MSort,
                     surfTar.xiSort, surfTar.yiSort, surfTar.ziSort,
                     surfTar.xcSort[surf], surfTar.ycSort[surf],
                     surfTar.zcSort[surf], index, param.P, param.kappa,
                     int(param.Nm), int(LorY))

    toc = time.time()
    timing.time_M2P += toc - tic

    return Ktx_aux, Kty_aux, Ktz_aux


def M2P_gpu(surfSrc, surfTar, K_gpu, V_gpu, surf, ind0, param, LorY, timing,
            kernel):
    """
    It computes the far field contribution of the double and single layer
    potential using the sorted data, on the GPU.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    K_gpu  : array, far field contribution to the double layer potential.
    V_gpu  : array, far field contribution to the single layer potential.
    ind0   : list, pointers to the location of the mulipole of order i,j,k
                   in the multipole array.
    param  : class, parameters related to the surface.
    LorY    int, Laplace (1) or Yukawa (2).
    timing : class, it contains timing information for different parts of
                    the code.
    kernel : pycuda source module.

    Returns
    --------
    K_gpu  : array, far field contribution to the double layer potential.
    V_gpu  : array, far field contribution to the single layer potential.
    """

    if param.GPU == 1:
        tic = cuda.Event()
        toc = cuda.Event()
    else:
        tic = Event()
        toc = Event()

    REAL = param.REAL

    tic.record()
    M2P_size = surfTar.offsetMlt[surf, len(surfTar.twig)]
    MSort = numpy.zeros(param.Nm * M2P_size)
    MdSort = numpy.zeros(param.Nm * M2P_size)

    i = -1
    for C in surfTar.M2P_list[surf, 0:M2P_size]:
        i += 1
        MSort[i * param.Nm:i * param.Nm + param.Nm] = surfSrc.tree[C].M
        MdSort[i * param.Nm:i * param.Nm + param.Nm] = surfSrc.tree[C].Md

    MDev = cuda.to_device(MSort.astype(REAL))
    MdDev = cuda.to_device(MdSort.astype(REAL))

    # GPU arrays are flattened, need to point to first element
    ptr_offset = surf * len(surfTar.offsetTwigs[surf]
                            )  # Pointer to first element of offset arrays
    ptr_list = surf * len(surfTar.P2P_list[surf]
                          )  # Pointer to first element in lists arrays

    GSZ = int(numpy.ceil(float(param.Nround) / param.NCRIT))  # CUDA grid size
    multipole_gpu = kernel.get_function("M2P")

    multipole_gpu(K_gpu,
                  V_gpu,
                  surfTar.offMltDev,
                  surfTar.sizeTarDev,
                  surfTar.xcDev,
                  surfTar.ycDev,
                  surfTar.zcDev,
                  MDev,
                  MdDev,
                  surfTar.xiDev,
                  surfTar.yiDev,
                  surfTar.ziDev,
                  ind0.indexDev,
                  numpy.int32(ptr_offset),
                  numpy.int32(ptr_list),
                  REAL(param.kappa),
                  numpy.int32(param.BlocksPerTwig),
                  numpy.int32(param.NCRIT),
                  numpy.int32(LorY),
                  block=(param.BSZ, 1, 1),
                  grid=(GSZ, 1))

    toc.record()
    toc.synchronize()
    timing.time_M2P += tic.time_till(toc) * 1e-3

    return K_gpu, V_gpu


def M2PKt_gpu(surfSrc, surfTar, Ktx_gpu, Kty_gpu, Ktz_gpu, surf, ind0, param,
              LorY, timing, kernel):
    """
    It computes the far field contribution of the adjoint double potential
    using the sorted data, on the GPU.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    Ktx_gpu: array, x component of the far field contribution to the adjoint
                    double layer potential.
    Kty_gpu: array, y component of the far field contribution to the adjoint
                    double layer potential.
    Ktz_gpu: array, z component of the far field contribution to the adjoint
                    double layer potential.
    surf   : int, position of the source surface in the surface array.
    ind0   : list, pointers to the location of the mulipole of order i,j,k
                   in the multipole array.
    param  : class, parameters related to the surface.
    LorY   : int, Laplace (1) or Yukawa (2).
    timing : class, it contains timing information for different parts of
                    the code.
    kernel : pycuda source module.

    Returns
    --------
    Ktx_gpu: array, x component of the far field contribution to the adjoint
                    double layer potential.
    Kty_gpu: array, y component of the far field contribution to the adjoint
                    double layer potential.
    Ktz_gpu: array, z component of the far field contribution to the adjoint
                    double layer potential.
    """

    if param.GPU == 1:
        tic = cuda.Event()
        toc = cuda.Event()
    else:
        tic = Event()
        toc = Event()

    REAL = param.REAL

    tic.record()
    M2P_size = surfTar.offsetMlt[surf, len(surfTar.twig)]
    MSort = numpy.zeros(param.Nm * M2P_size)

    i = -1
    for C in surfTar.M2P_list[surf, 0:M2P_size]:
        i += 1
        MSort[i * param.Nm:i * param.Nm + param.Nm] = surfSrc.tree[C].M

    MDev = cuda.to_device(MSort.astype(REAL))

    # GPU arrays are flattened, need to point to first element
    ptr_offset = surf * len(surfTar.offsetTwigs[surf]
                            )  # Pointer to first element of offset arrays
    ptr_list = surf * len(surfTar.P2P_list[surf]
                          )  # Pointer to first element in lists arrays

    GSZ = int(numpy.ceil(float(param.Nround) / param.NCRIT))  # CUDA grid size
    multipoleKt_gpu = kernel.get_function("M2PKt")

    multipoleKt_gpu(Ktx_gpu,
                    Kty_gpu,
                    Ktz_gpu,
                    surfTar.offMltDev,
                    surfTar.sizeTarDev,
                    surfTar.xcDev,
                    surfTar.ycDev,
                    surfTar.zcDev,
                    MDev,
                    surfTar.xiDev,
                    surfTar.yiDev,
                    surfTar.ziDev,
                    ind0.indexDev,
                    numpy.int32(ptr_offset),
                    numpy.int32(ptr_list),
                    REAL(param.kappa),
                    numpy.int32(param.BlocksPerTwig),
                    numpy.int32(param.NCRIT),
                    numpy.int32(LorY),
                    block=(param.BSZ, 1, 1),
                    grid=(GSZ, 1))

    toc.record()
    toc.synchronize()
    timing.time_M2P += tic.time_till(toc) * 1e-3

    return Ktx_gpu, Kty_gpu, Ktz_gpu


def P2P_sort(surfSrc, surfTar, m, mx, my, mz, mKc, mVc, K_aux, V_aux, surf,
             LorY, K_diag, V_diag, IorE, L, w, param, timing):
    """
    It computes the near field contribution of the double and single layer
    potential using the sorted data and adds it to the far field contribution
    given as an input.

    Note: In this context when we refer to mass we mean
                 mass       = (vector x gauss weights)
                 mass-clean = (vector)

          where 'vector' is the vector in the matrix-vector multiplication in
          the GMRES.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    m      : array, mass of the source particle for the single layer potential
                    calculation.
    mx     : array, mass of the source particle times  the 'x' component of the
                    normal vector, for the double layer potential calculation.
    my     : array, mass of the source particle times  the 'y' component of the
                    normal vector, for the double layer potential calculation.
    mz     : array, mass of the source particle times  the 'z' component of the
                    normal vector, for the double layer potential calculation.
    mKc    : array, mass-clean of the source particle for the double layer
                    potential calculation.
    mVc    : array, mass-clean of the source particle for the double layer
                    potential calculation.
    K_aux  : array, far field contribution to the double layer potential.
    V_aux  : array, far field contribution to the single layer potential.
    surf   : int, position of the source surface in the surface array.
    K_diag : array, diagonal elements of the double layer integral operator.
    V_diag : array, diagonal elements of the single layer integral operator.
    IorE   : int, internal (1) or external (2).
    L      : float, representative distance of the triangles. (sqrt{2*Area})
    w      : array, gauss points.
    param  : class, parameters related to the surface.
    timing : class, it contains timing information for different parts of
                    the code.

    Returns
    --------
    K_aux  : array, far plus near field contribution to the double layer
                    potential.
    V_aux  : array, far plus near field contribution to the single layer
                    potential.
    """

    tic = time.time()

    s_xj = surfSrc.xjSort
    s_yj = surfSrc.yjSort
    s_zj = surfSrc.zjSort

    xt = surfTar.xiSort
    yt = surfTar.yiSort
    zt = surfTar.ziSort

    tri = surfSrc.sortSource / param.K  # Triangle
    k = surfSrc.sortSource % param.K  # Gauss point

    aux = numpy.zeros(2)

    direct_sort(
        K_aux, V_aux, int(LorY), K_diag, V_diag, int(IorE),
        numpy.ravel(surfSrc.vertex[surfSrc.triangleSort[:]]), numpy.int32(tri),
        numpy.int32(k), surfTar.xi, surfTar.yi, surfTar.zi, s_xj, s_yj, s_zj,
        xt, yt, zt, m, mx, my, mz, mKc, mVc, surfTar.P2P_list[surf],
        surfTar.offsetTarget, surfTar.sizeTarget, surfSrc.offsetSource,
        surfTar.offsetTwigs[surf], numpy.int32(surfTar.tree[0].target),
        surfSrc.AreaSort, surfSrc.sglInt_intSort, surfSrc.sglInt_extSort,
        surfSrc.xk, surfSrc.wk, surfSrc.Xsk, surfSrc.Wsk, param.kappa,
        param.threshold, param.eps, w[0], aux)

    timing.AI_int += int(aux[0])
    timing.time_an += aux[1]

    toc = time.time()
    timing.time_P2P += toc - tic

    return K_aux, V_aux


def P2PKt_sort(surfSrc, surfTar, m, mKc, Ktx_aux, Kty_aux, Ktz_aux, surf, LorY,
               w, param, timing):
    """
    It computes the near field contribution of the double and single layer
    potential using the sorted data and adds it to the far field contribution
    given as an input.

    Note: In this context when we refer to mass we mean
                 mass       = (vector x gauss weights)
                 mass-clean = (vector)

          where 'vector' is the vector in the matrix-vector multiplication in
          the GMRES.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    m      : array, mass of the source particle for the adjoint double layer
                    potential calculation.
    mKc    : array, mass-clean of the source particle for the adjoint double
                    layer potential calculation.
    Ktx_aux: array, x component of the far field contribution to the adjoint
                    double layer potential.
    Kty_aux: array, y component of the far field contribution to the adjoint
                    double layer potential.
    Ktz_aux: array, z component of the far field contribution to the adjoint
                    double layer potential.
    surf   : int, position of the source surface in the surface array.
    LorY   : int, Laplace (1) or Yukawa (2).
    w      : array, gauss points.
    param  : class, parameters related to the surface.
    timing : class, it contains timing information for different parts of
                    the code.

    Returns
    --------
    Ktx_aux: array, x component of the far plus near field contribution to the
                    adjoint double layer potential.
    Kty_aux: array, y component of the far plus near field contribution to the
                    adjoint double layer potential.
    Ktz_aux: array, z component of the far plus near field contribution to the
                    adjoint double layer potential.
    """

    tic = time.time()

    s_xj = surfSrc.xjSort
    s_yj = surfSrc.yjSort
    s_zj = surfSrc.zjSort

    xt = surfTar.xiSort
    yt = surfTar.yiSort
    zt = surfTar.ziSort

    k = surfSrc.sortSource % param.K  # Gauss point

    aux = numpy.zeros(2)

    directKt_sort(Ktx_aux, Kty_aux, Ktz_aux, int(LorY),
                  numpy.ravel(surfSrc.vertex[surfSrc.triangleSort[:]]),
                  numpy.int32(k), s_xj, s_yj, s_zj, xt, yt, zt, m, mKc,
                  surfTar.P2P_list[surf], surfTar.offsetTarget,
                  surfTar.sizeTarget, surfSrc.offsetSource,
                  surfTar.offsetTwigs[surf], surfSrc.AreaSort, surfSrc.Xsk,
                  surfSrc.Wsk, param.kappa, param.threshold, param.eps, aux)

    timing.AI_int += int(aux[0])
    timing.time_an += aux[1]

    toc = time.time()
    timing.time_P2P += toc - tic

    return Ktx_aux, Kty_aux, Ktz_aux


def P2P_gpu(surfSrc, surfTar, m, mx, my, mz, mKc, mVc, K_gpu, V_gpu, surf,
            LorY, K_diag, IorE, L, w, param, timing, kernel):
    """
    It computes the near field contribution of the double and single layer
    potential using the sorted data and adds it to the far field contribution
    given as an input, on the GPU.

    Note: In this context when we refer to mass we mean
                 mass       = (vector x gauss weights)
                 mass-clean = (vector)

          where 'vector' is the vector in the matrix-vector multiplication in
          the GMRES.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    m      : array, mass of the source particle for the single layer potential
                    calculation.
    mx     : array, mass of the source particle times  the 'x' component of the
                    normal vector, for the double layer potential calculation.
    my     : array, mass of the source particle times  the 'y' component of the
                    normal vector, for the double layer potential calculation.
    mz     : array, mass of the source particle times  the 'z' component of the
                    normal vector, for the double layer potential calculation.
    mKc    : array, mass-clean of the source particle for the double layer
                    potential calculation.
    mVc    : array, mass-clean of the source particle for the double layer
                    potential calculation.
    K_gpu  : array, far field contribution to the double layer potential.
    V_gpu  : array, far field contribution to the single layer potential.
    surf   : int, position of the source surface in the surface array.
    K_diag : array, diagonal elements of the double layer integral operator.
    IorE   : int, internal (1) or external (2).
    L      : float, representative distance of the triangles. (sqrt{2*Area})
    w      : array, gauss points.
    param  : class, parameters related to the surface.
    timing : class, it contains timing information for different parts of
                    the code.
    kernel : pycuda source module.

    Returns
    --------
    K_gpu  : array, far plus near field contribution to the double layer
                    potential.
    V_gpu  : array, far plus near field contribution to the single layer
                    potential.
    """

    if param.GPU == 1:
        tic = cuda.Event()
        toc = cuda.Event()
    else:
        tic = Event()
        toc = Event()

    tic.record()
    REAL = param.REAL
    mDev = cuda.to_device(m.astype(REAL))
    mxDev = cuda.to_device(mx.astype(REAL))
    myDev = cuda.to_device(my.astype(REAL))
    mzDev = cuda.to_device(mz.astype(REAL))
    mKcDev = cuda.to_device(mKc.astype(REAL))
    mVcDev = cuda.to_device(mVc.astype(REAL))
    toc.record()
    toc.synchronize()
    timing.time_trans += tic.time_till(toc) * 1e-3

    tic.record()
    GSZ = int(numpy.ceil(float(param.Nround) / param.NCRIT))  # CUDA grid size
    direct_gpu = kernel.get_function("P2P")
    AI_int = cuda.to_device(numpy.zeros(param.Nround, dtype=numpy.int32))

    # GPU arrays are flattened, need to point to first element
    ptr_offset = surf * len(surfTar.offsetTwigs[surf]
                            )  # Pointer to first element of offset arrays
    ptr_list = surf * len(surfTar.P2P_list[surf]
                          )  # Pointer to first element in lists arrays

    # Check if internal or external to send correct singular integral
    if IorE == 1:
        sglInt = surfSrc.sglInt_intDev
    else:
        sglInt = surfSrc.sglInt_extDev

    direct_gpu(K_gpu,
               V_gpu,
               surfSrc.offSrcDev,
               surfTar.offTwgDev,
               surfTar.P2P_lstDev,
               surfTar.sizeTarDev,
               surfSrc.kDev,
               surfSrc.xjDev,
               surfSrc.yjDev,
               surfSrc.zjDev,
               mDev,
               mxDev,
               myDev,
               mzDev,
               mKcDev,
               mVcDev,
               surfTar.xiDev,
               surfTar.yiDev,
               surfTar.ziDev,
               surfSrc.AreaDev,
               sglInt,
               surfSrc.vertexDev,
               numpy.int32(ptr_offset),
               numpy.int32(ptr_list),
               numpy.int32(LorY),
               REAL(param.kappa),
               REAL(param.threshold),
               numpy.int32(param.BlocksPerTwig),
               numpy.int32(param.NCRIT),
               REAL(K_diag),
               AI_int,
               surfSrc.XskDev,
               surfSrc.WskDev,
               block=(param.BSZ, 1, 1),
               grid=(GSZ, 1))

    toc.record()
    toc.synchronize()
    timing.time_P2P += tic.time_till(toc) * 1e-3

    tic.record()
    AI_aux = numpy.zeros(param.Nround, dtype=numpy.int32)
    AI_aux = cuda.from_device(AI_int, param.Nround, dtype=numpy.int32)
    timing.AI_int += sum(AI_aux[surfTar.unsort])
    toc.record()
    toc.synchronize()
    timing.time_trans += tic.time_till(toc) * 1e-3

    return K_gpu, V_gpu


def P2PKt_gpu(surfSrc, surfTar, m, mKtc, Ktx_gpu, Kty_gpu, Ktz_gpu, surf, LorY,
              w, param, timing, kernel):
    """
    It computes the near field contribution of the double and single layer
    potential using the sorted data and adds it to the far field contribution
    given as an input, on the GPU.

    Note: In this context when we refer to mass we mean

                 mass       = (vector x gauss weights)
                 mass-clean = (vector)

          where 'vector' is the vector in the matrix-vector multiplication in
          the GMRES.

    Arguments
    ----------
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    m      : array, mass of the source particle for the adjoint double layer
                    potential calculation.
    mKc    : array, mass-clean of the source particle for the adjoint double
                    layer potential calculation.
    Ktx_gpu: array, x component of the far field contribution to the adjoint
                    double layer potential.
    Kty_gpu: array, y component of the far field contribution to the adjoint
                    double layer potential.
    Ktz_gpu: array, z component of the far field contribution to the adjoint
                    double layer potential.
    surf   : int, position of the source surface in the surface array.
    LorY   : int, Laplace (1) or Yukawa (2).
    w      : array, gauss points.
    param  : class, parameters related to the surface.
    timing : class, it contains timing information for different parts of
                    the code.
    kernel : pycuda source module.

    Returns
    --------
    Ktx_gpu: array, x component of the far plus near field contribution to the
                    adjoint double layer potential.
    Kty_gpu: array, y component of the far plus near field contribution to the
                    adjoint double layer potential.
    Ktz_gpu: array, z component of the far plus near field contribution to the
                    adjoint double layer potential.

    """

    if param.GPU == 1:
        tic = cuda.Event()
        toc = cuda.Event()
    else:
        tic = Event()
        toc = Event()

    tic.record()
    REAL = param.REAL
    mDev = cuda.to_device(m.astype(REAL))
    mKtcDev = cuda.to_device(mKtc.astype(REAL))
    toc.record()
    toc.synchronize()
    timing.time_trans += tic.time_till(toc) * 1e-3

    tic.record()
    GSZ = int(numpy.ceil(float(param.Nround) / param.NCRIT))  # CUDA grid size
    directKt_gpu = kernel.get_function("P2PKt")
    AI_int = cuda.to_device(numpy.zeros(param.Nround, dtype=numpy.int32))

    # GPU arrays are flattened, need to point to first element
    ptr_offset = surf * len(surfTar.offsetTwigs[surf]
                            )  # Pointer to first element of offset arrays
    ptr_list = surf * len(surfTar.P2P_list[surf]
                          )  # Pointer to first element in lists arrays

    directKt_gpu(Ktx_gpu,
                 Kty_gpu,
                 Ktz_gpu,
                 surfSrc.offSrcDev,
                 surfTar.offTwgDev,
                 surfTar.P2P_lstDev,
                 surfTar.sizeTarDev,
                 surfSrc.kDev,
                 surfSrc.xjDev,
                 surfSrc.yjDev,
                 surfSrc.zjDev,
                 mDev,
                 mKtcDev,
                 surfTar.xiDev,
                 surfTar.yiDev,
                 surfTar.ziDev,
                 surfSrc.AreaDev,
                 surfSrc.vertexDev,
                 numpy.int32(ptr_offset),
                 numpy.int32(ptr_list),
                 numpy.int32(LorY),
                 REAL(param.kappa),
                 REAL(param.threshold),
                 numpy.int32(param.BlocksPerTwig),
                 numpy.int32(param.NCRIT),
                 AI_int,
                 surfSrc.XskDev,
                 surfSrc.WskDev,
                 block=(param.BSZ, 1, 1),
                 grid=(GSZ, 1))

    toc.record()
    toc.synchronize()
    timing.time_P2P += tic.time_till(toc) * 1e-3

    tic.record()
    AI_aux = numpy.zeros(param.Nround, dtype=numpy.int32)
    AI_aux = cuda.from_device(AI_int, param.Nround, dtype=numpy.int32)
    timing.AI_int += sum(AI_aux[surfTar.unsort])
    toc.record()
    toc.synchronize()
    timing.time_trans += tic.time_till(toc) * 1e-3

    return Ktx_gpu, Kty_gpu, Ktz_gpu


def M2P_nonvec(Cells, CJ, xq, Kval, Vval, index, par_reac, source, time_M2P):
    """
    It computes the far field contribution of the double and single layer
    potential without doing the assumption that all the particles in the same
    twig cell have the same interaction list.

    This is used for the calculation for the reaction potential where the
    targets are the point-charges location.

    Arguments
    ----------
    Cells   : array, cells of the tree.
    CJ      : int, index of the source cell.
    xq      : array, postion of the point charges.
    Kval  : array, far field contribution to the double layer potential.
    Vval  : array, far field contribution to the single layer potential.
    index   : list, pointers to the location of the mulipole of order i,j,k
                    in the multipole array.
    par_reac: class, fine parameters related to the surface.
    source  : list, P2P interaction list, which is a list of the cells that
                    each charge-point interacts by P2P.
    time_M2P: real, timed consumed in compute M2P_nonvec function.

    Returns
    --------
    Kval  : array, far field contribution to the double layer potential.
    Vval  : array, far field contribution to the single layer potential.
    source  : list, P2P interaction list, which is a list of the cells that
                    each charge-point interacts.
    time_M2P: real, time consumed in compute M2P_nonvec function.
    """

    if (Cells[CJ].ntarget >= par_reac.NCRIT):  # if not a twig
        for c in range(8):
            if (Cells[CJ].nchild & (1 << c)):
                CC = Cells[CJ].child[c]  # Points at child cell
                dxi = Cells[CC].xc - xq[0]
                dyi = Cells[CC].yc - xq[1]
                dzi = Cells[CC].zc - xq[2]
                r = numpy.sqrt(dxi * dxi + dyi * dyi + dzi * dzi)
                if Cells[
                        CC].r > par_reac.theta * r:  # Max distance between particles
                    Kval, Vval, source, time_M2P = M2P_nonvec(
                        Cells, CC, xq, Kval, Vval, index, par_reac, source,
                        time_M2P)
                else:
                    tic = time.time()
                    dxi = xq[0] - Cells[CC].xc
                    dyi = xq[1] - Cells[CC].yc
                    dzi = xq[2] - Cells[CC].zc

                    K_aux = numpy.zeros(1)
                    V_aux = numpy.zeros(1)
                    dxi = numpy.array([dxi])
                    dyi = numpy.array([dyi])
                    dzi = numpy.array([dzi])
                    LorY = 1
                    multipole_c(K_aux, V_aux, Cells[CC].M, Cells[CC].Md, dxi,
                                dyi, dzi, index, par_reac.P, par_reac.kappa,
                                int(par_reac.Nm), int(LorY))

                    Kval += K_aux
                    Vval += V_aux
                    toc = time.time()
                    time_M2P += toc - tic

    else:  # Else on a twig cell
        source.extend(Cells[CJ].source)

    return Kval, Vval, source, time_M2P


def P2P_nonvec(Cells, surface, m, mx, my, mz, mKc, mVc, xq, Kval, Vval, IorE,
               par_reac, w, source, AI_int, time_P2P):
    """
    It computes the near field contribution of the double and single layer
    potential and adds it to the far field contribution given as an input.
    In this case we don't do the assumption that all the particles in the same
    twig cell have the same interaction list.

    This is used for the calculation for the reaction potential where the
    targets are the point-charges location.

    Arguments
    ----------
    Cells   : array, cells of the tree.
    surface : class, surface where we are computing the P2P_nonvec.
    m       : array, mass of the source particle for the single layer potential
                    calculation.
    mx      : array, mass of the source particle times  the 'x' component of the
                    normal vector, for the double layer potential calculation.
    my      : array, mass of the source particle times  the 'y' component of the
                    normal vector, for the double layer potential calculation.
    mz      : array, mass of the source particle times  the 'z' component of the
                    normal vector, for the double layer potential calculation.
    mKc     : array, mass-clean of the source particle for the double layer
                    potential calculation.
    mVc     : array, mass-clean of the source particle for the double layer
                    potential calculation.
    xq      : array, postion of the point charges.
    Kval  : array, far field contribution to the double layer potential.
    Vval  : array, far field contribution to the single layer potential.
    IorE    : int, internal (1) or external (2).
    par_reac: class, fine parameters related to the surface.
    w       : array, gauss points.
    source  : list, P2P interaction list, which is a list of the cells that
                    each charge-point interacts.
    AI_int  : int, counter of the amount of near singular integrals solved.
    time_P2P: real, timed consumed in compute P2P_nonvec function.

    Returns
    --------
    Kval   : array, far plus near field contribution to the double layer
                    potential.
    Vval  : array, far plus near field contribution to the single layer
                    potential.
    AI_int  : int, counter of the amount of near singular integrals solved.
    time_P2P: real, timed consumed in compute P2P_nonvec function.
    """

    tic = time.time()
    LorY = 1
    source = numpy.int32(numpy.array(source))
    s_xj = surface.xj[source]
    s_yj = surface.yj[source]
    s_zj = surface.zj[source]
    s_m = m[source]
    s_mx = mx[source]
    s_my = my[source]
    s_mz = mz[source]
    s_mKc = mKc[source]
    s_mVc = mVc[source]

    tri = source / par_reac.K  # Triangle
    k = source % par_reac.K  # Gauss point

    K_aux = numpy.zeros(1)
    V_aux = numpy.zeros(1)

    xq_arr = numpy.array([xq[0]])
    yq_arr = numpy.array([xq[1]])
    zq_arr = numpy.array([xq[2]])

    aux = numpy.zeros(2)
    K_diag = 0
    V_diag = 0
    direct_c(K_aux,
             V_aux,
             int(LorY),
             K_diag,
             V_diag,
             int(IorE),
             numpy.ravel(surface.vertex[surface.triangle[:]]),
             numpy.int32(tri),
             numpy.int32(k),
             surface.xi,
             surface.yi,
             surface.zi,
             s_xj,
             s_yj,
             s_zj,
             xq_arr,
             yq_arr,
             zq_arr,
             s_m,
             s_mx,
             s_my,
             s_mz,
             s_mKc,
             s_mVc,
             numpy.array(
                 [-1], dtype=numpy.int32),
             surface.area,
             surface.sglInt_int,
             surface.sglInt_ext,
             surface.xk,
             surface.wk,
             surface.Xsk,
             surface.Wsk,
             par_reac.kappa,
             par_reac.threshold,
             par_reac.eps,
             w[0],
             aux)

    AI_int += int(aux[0])

    Kval += K_aux
    Vval += V_aux
    toc = time.time()
    time_P2P += toc - tic

    return Kval, Vval, AI_int, time_P2P
