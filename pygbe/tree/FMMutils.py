"""
It contains the functions to build the tree and 
"""
import numpy
from scipy.misc import factorial, comb

# Wrapped code
from multipole import multipole_c, setIndex, getIndex_arr, multipole_sort, multipoleKt_sort
from direct import direct_c, direct_sort, directKt_sort
from calculateMultipoles import P2M, M2M

# CUDA libraries
try:
    import pycuda.driver as cuda
except:
    print('PyCUDA not installed, performance might not be so hot')

import time


class Cell():
    """
    Cell class. It contains the information about the cells in the tree.

    Attributes:
    -----------
    nsource   : int, Number of source particles.
    ntarget   : int, Number of target particles.
    nchild    : int, Number of child boxes in binary, 8bit value, if certain
                     child exists, that bit will be 1.
    source    : array, Pointer to source particles.
    target    : array, Pointer to target particles.
    xc        : float, x position of cell.
    yc        : float, y position of cell.
    zc        : float, z position of cell.
    r         : float, cell radius.
    parent    : int, Pointer to parent cell.
    child     : array, Pointer to child cell.
    M         : array, Array with multipoles.
    Md        : array, Array with multipoles for grad(G).n.
    P2P_list  : list, Pointer to cells that interact with P2P.
    M2P_list  : list, Pointer to cells that interact with M2P.
    M2P_size  : list, Size of the M2P interaction list.
    list_ready: int, Flag to know if P2P list is already generated.
    twig_array: list, Position in the twig array.
     
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

    Arguments:
    ----------
    octant: int, octant of the child cell. 
    Cells : array, it contains the cells information.
    i     : int, index of parent cell in Cells array.
    NCRIT : int, maximum number of boundary elements per twig box of tree
                    structure.
    Nm    : int, number of multipole coefficients.
    Ncell :

    Returns:
    --------
    Ncell :
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
    It splits a cell with more than NCRIT particles.

    Arguments:
    ----------
    x    :
    y    :
    z    : 
    Cells: array, it contains the cells information.
    C    : int, index in the Cells array of the cell to be splited .
    NCRIT: int, maximum number of boundary elements per twig box of tree
                    structure.
    Nm   : int, number of multipole coefficients.
    Ncell:

    Returns:
    --------
    Ncell:

    """
    
    # split_cell splits cell with more than NCRIT particles
    # x,y,z: positions of particles
    # Cells: array of cells
    # C    : index of cell to be split in Cells array

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
    # Target-based tree

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
    # Cells     : array of cells
    # C         : index of cell in Cells array 
    # twig      : array with indices of twigs in Cells array

    if (Cells[C].ntarget >= NCRIT):
        for c in range(8):
            if (Cells[C].nchild & (1 << c)):
                twig = findTwigs(Cells, Cells[C].child[c], twig, NCRIT)
    else:
        twig.append(C)
        Cells[C].twig_array = numpy.int32(len(twig) - 1)

    return twig


def addSources2(x, y, z, j, Cells, C, NCRIT):
    # x,y,z: location of sources
    # j    : index of source
    # Cells: array with cells
    # C    : index of cell in Cells array
    # NCRIT: max number of target particles per cell

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


def addSources(x, y, z, Cells, twig):
    # x,y,z: location of sources
    # Cells: array with cells
    # twig : array with pointers to twigs of cells array

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

    close_twig = argmin(r, axis=0)

    for j in range(len(close_twig)):
        Cells[twig[close_twig[j]]].nsource += 1
        Cells[twig[close_twig[j]]].source = numpy.append(
            Cells[twig[close_twig[j]]].source, j)


def addSources3(Cells, twig, K):
    # This version of addSources puts the sources in the same cell
    # as the collocation point of the same panel
    for C in twig:
        Cells[C].nsource = K * Cells[C].ntarget
        for j in range(K):
            Cells[C].source = numpy.append(Cells[C].source,
                                           K * Cells[C].target + j)


def sortPoints(surface, Cells, twig, param):

    Nround = len(twig) * param.NCRIT

    surface.sortTarget = numpy.zeros(Nround, dtype=numpy.int32)
    surface.unsort = numpy.zeros(len(surface.xi), dtype=numpy.int32)
    surface.sortSource = numpy.zeros(len(surface.xj), dtype=numpy.int32)
    surface.offsetSource = numpy.zeros(len(twig) + 1, dtype=numpy.int32)
    surface.offsetTarget = numpy.zeros(len(twig), dtype=numpy.int32)
    surface.sizeTarget = numpy.zeros(len(twig), dtype=numpy.int32)
    offTar = 0
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
    surface.AreaSort = surface.Area[surface.sortSource / param.K]
    surface.sglInt_intSort = surface.sglInt_int[surface.sortSource / param.K]
    surface.sglInt_extSort = surface.sglInt_ext[surface.sortSource / param.K]
    surface.triangleSort = surface.triangle[surface.sortSource / param.K]


def computeIndices(P, ind0):
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
    # Cells     : array of Cells
    # CJ        : index of source cell
    # CI        : index of target cell
    # theta     : MAC criteron 
    # NCRIT     : max number of particles per cell

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
                    I = surfTar.tree[CI].M2P_size[s_src]
                    surfTar.M2P_list[s_src, offMlt] = CC
                    offMlt += 1
    else:
        twig_cell = surfSrc.tree[CJ].twig_array
        surfTar.P2P_list[s_src, offTwg] = twig_cell
        offTwg += 1

    return offTwg, offMlt


def generateList(surf_array, field_array, param):
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
    # Cells     : array of cells
    # C         : index of cell in Cells array
    # x,y,z     : position of particles
    # m         : weight of particles
    # P         : order of Taylor expansion
    # NCRIT     : max number of target particles per cell
    # II,JJ,KK  : x,y,z powers of multipole expansion
    # index     : 1D mapping of II,JJ,KK (index of multipoles)

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
    # Cells     : array of cells
    # CC        : index of child cell in Cells array
    # PC        : index of parent cell in Cells array
    # P         : order of Taylor expansion
    # II,JJ,KK  : x,y,z powers of multipole expansion
    # index     : 1D mapping of II,JJ,KK (index of multipoles)

    dx = Cells[PC].xc - Cells[CC].xc
    dy = Cells[PC].yc - Cells[CC].yc
    dz = Cells[PC].zc - Cells[CC].zc

    M2M(Cells[PC].M, Cells[CC].M, dx, dy, dz, II, JJ, KK, combII, combJJ,
        combKK, IImii, JJmjj, KKmkk, index_small, index_ptr)
    M2M(Cells[PC].Md, Cells[CC].Md, dx, dy, dz, II, JJ, KK, combII, combJJ,
        combKK, IImii, JJmjj, KKmkk, index_small, index_ptr)


def M2P_sort(surfSrc, surfTar, K_aux, V_aux, surf, index, param, LorY, timing):

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

    tic = time.time()
    M2P_size = surfTar.offsetMlt[surf, len(surfTar.twig)]
    MSort = numpy.zeros(param.Nm * M2P_size)
    MdSort = numpy.zeros(param.Nm * M2P_size)

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

#    (free, total) = cuda.mem_get_info()
#    print 'Global memory occupancy: %f%% free'%(free*100/total)
    MDev = cuda.to_device(MSort.astype(REAL))
    MdDev = cuda.to_device(MdSort.astype(REAL))
    #    (free, total) = cuda.mem_get_info()
    #    print 'Global memory occupancy: %f%% free'%(free*100/total)

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

#    (free, total) = cuda.mem_get_info()
#    print 'Global memory occupancy: %f%% free'%(free*100/total)
    MDev = cuda.to_device(MSort.astype(REAL))
    #    (free, total) = cuda.mem_get_info()
    #    print 'Global memory occupancy: %f%% free'%(free*100/total)

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
    # Cells     : array of Cells
    # CJ        : index of source cell
    # p         : accumulator 
    # theta     : MAC criteron 
    # Nm        : Number of multipole coefficients
    # P         : order of Taylor expansion
    # NCRIT     : max number of particles per cell

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
    target = numpy.array([-1], dtype=numpy.int32)

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
             surface.Area,
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
