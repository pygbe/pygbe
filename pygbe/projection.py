"""
It contains the functions to calculate the different potentials:
-The single and double layer potential.
-The adjoint double layer potential.
-The reaction potential.
"""
import numpy
from numpy import pi

from pygbe.classes import Event
from pygbe.quadrature import getWeights
from pygbe.tree.FMMutils import (getMultipole, upwardSweep, M2P_sort, M2PKt_sort,
                                 M2P_gpu, M2PKt_gpu, P2P_sort, P2PKt_sort, P2P_gpu,
                                 P2PKt_gpu, M2P_nonvec, P2P_nonvec)

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
except:
    pass

import time


def project(XK, XV, LorY, surfSrc, surfTar, K_diag, V_diag, IorE, self, param,
            ind0, timing, kernel):
    """
    It computes the single and double layer potentials.

    Arguments
    ----------
    XK     : array, input for the double layer potential.
    XV     : array, input for the single layer potential.
    LorY   : int, Laplace (1) or Yukawa (2).
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation
                    points.
    K_diag : array, diagonal elements of the double layer integral operator.
    V_diag : array, diagonal elements of the single layer integral operator.
    IorE   : int, internal (1) or external (2).
    self   : int, position in the surface array of the source surface.
    param  : class, parameters related to the surface.
    ind0   : array, it contains the indices related to the treecode computation.
    timing : class, it contains timing information for different parts of the
                    code.
    kernel : pycuda source module.

    Returns
    --------
    K_lyr  : array, double layer potential.
    V_lyr  : array, single layer potential.
    """

    if param.GPU == 1:
        tic = cuda.Event()
        toc = cuda.Event()
    else:
        tic = Event()
        toc = Event()

    REAL = param.REAL
    Ns = len(surfSrc.triangle)
    L = numpy.sqrt(2 * surfSrc.area)  # Representative length

    tic.record()
    K = param.K
    w = getWeights(K)
    X_V = numpy.zeros(Ns * K)
    X_Kx = numpy.zeros(Ns * K)
    X_Ky = numpy.zeros(Ns * K)
    X_Kz = numpy.zeros(Ns * K)
    X_Kc = numpy.zeros(Ns * K)
    X_Vc = numpy.zeros(Ns * K)

    NsK = numpy.arange(Ns * K)
    X_V[:] = XV[NsK // K] * w[NsK % K] * surfSrc.area[NsK // K]
    X_Kx[:] = XK[NsK // K] * w[NsK % K] * surfSrc.area[
        NsK // K] * surfSrc.normal[NsK // K, 0]
    X_Ky[:] = XK[NsK // K] * w[NsK % K] * surfSrc.area[
        NsK // K] * surfSrc.normal[NsK // K, 1]
    X_Kz[:] = XK[NsK // K] * w[NsK % K] * surfSrc.area[
        NsK // K] * surfSrc.normal[NsK // K, 2]
    X_Kc[:] = XK[NsK // K]
    X_Vc[:] = XV[NsK // K]

    toc.record()
    toc.synchronize()
    timing.time_mass += tic.time_till(toc) * 1e-3

    tic.record()
    C = 0
    getMultipole(surfSrc.tree, C, surfSrc.xj, surfSrc.yj, surfSrc.zj, X_V,
                 X_Kx, X_Ky, X_Kz, ind0, param.P, param.NCRIT)
    toc.record()
    toc.synchronize()
    timing.time_P2M += tic.time_till(toc) * 1e-3

    tic.record()
    for C in reversed(range(1, len(surfSrc.tree))):
        PC = surfSrc.tree[C].parent
        upwardSweep(surfSrc.tree, C, PC, param.P, ind0.II, ind0.JJ, ind0.KK,
                    ind0.index, ind0.combII, ind0.combJJ, ind0.combKK,
                    ind0.IImii, ind0.JJmjj, ind0.KKmkk, ind0.index_small,
                    ind0.index_ptr)
    toc.record()
    toc.synchronize()
    timing.time_M2M += tic.time_till(toc) * 1e-3

    tic.record()
    X_V = X_V[surfSrc.sortSource]
    X_Kx = X_Kx[surfSrc.sortSource]
    X_Ky = X_Ky[surfSrc.sortSource]
    X_Kz = X_Kz[surfSrc.sortSource]
    X_Kc = X_Kc[surfSrc.sortSource]
    X_Vc = X_Vc[surfSrc.sortSource]
    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc) * 1e-3

    param.Nround = len(surfTar.twig) * param.NCRIT
    K_aux = numpy.zeros(param.Nround)
    V_aux = numpy.zeros(param.Nround)

    ### CPU code
    if param.GPU == 0:
        K_aux, V_aux = M2P_sort(surfSrc, surfTar, K_aux, V_aux, self,
                                ind0.index_large, param, LorY, timing)

        K_aux, V_aux = P2P_sort(surfSrc, surfTar, X_V, X_Kx, X_Ky, X_Kz, X_Kc,
                                X_Vc, K_aux, V_aux, self, LorY, K_diag, V_diag,
                                IorE, L, w, param, timing)

    ### GPU code
    elif param.GPU == 1:
        K_gpu = cuda.to_device(K_aux.astype(REAL))
        V_gpu = cuda.to_device(V_aux.astype(REAL))

        if surfTar.offsetMlt[self, len(surfTar.twig)] > 0:
            K_gpu, V_gpu = M2P_gpu(surfSrc, surfTar, K_gpu, V_gpu, self, ind0,
                                   param, LorY, timing, kernel)

        K_gpu, V_gpu = P2P_gpu(surfSrc, surfTar, X_V, X_Kx, X_Ky, X_Kz, X_Kc,
                               X_Vc, K_gpu, V_gpu, self, LorY, K_diag, IorE, L,
                               w, param, timing, kernel)

        tic.record()
        K_aux = cuda.from_device(K_gpu, len(K_aux), dtype=REAL)
        V_aux = cuda.from_device(V_gpu, len(V_aux), dtype=REAL)
        toc.record()
        toc.synchronize()
        timing.time_trans += tic.time_till(toc) * 1e-3

    tic.record()
    K_lyr = K_aux[surfTar.unsort]
    V_lyr = V_aux[surfTar.unsort]
    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc) * 1e-3

    return K_lyr, V_lyr


def project_Kt(XKt, LorY, surfSrc, surfTar, Kt_diag, self, param, ind0, timing,
               kernel):
    """
    It computes the adjoint double layer potential.

    Arguments
    ----------
    XKt    : array, input for the adjoint double layer potential.
    LorY   : int, Laplace (1) or Yukawa (2).
    surfSrc: class, source surface, the one that contains the gauss points.
    surfTar: class, target surface, the one that contains the collocation points.
    Kt_diag: array, diagonal elements of the adjoint double layer integral
                    operator.
    self   : int, position in the surface array of the source surface.
    param  : class, parameters related to the surface.
    ind0   : array, it contains the indices related to the treecode computation.
    timing : class, it contains timing information for different parts of the
                    code.
    kernel : pycuda source module.

    Returns
    --------
    Kt_lyr: array, adjoint double layer potential.
    """

    if param.GPU == 1:
        tic = cuda.Event()
        toc = cuda.Event()
    else:
        tic = Event()
        toc = Event()

    REAL = param.REAL
    Ns = len(surfSrc.triangle)

    tic.record()
    K = param.K
    w = getWeights(K)
    X_Kt = numpy.zeros(Ns * K)
    X_Ktc = numpy.zeros(Ns * K)

    NsK = numpy.arange(Ns * K)
    X_Kt[:] = XKt[NsK // K] * w[NsK % K] * surfSrc.area[NsK // K]
    X_Ktc[:] = XKt[NsK // K]

    toc.record()
    toc.synchronize()
    timing.time_mass += tic.time_till(toc) * 1e-3

    tic.record()
    C = 0
    X_aux = numpy.zeros(Ns * K)
    getMultipole(surfSrc.tree, C, surfSrc.xj, surfSrc.yj, surfSrc.zj, X_Kt,
                 X_aux, X_aux, X_aux, ind0, param.P, param.NCRIT)
    toc.record()
    toc.synchronize()
    timing.time_P2M += tic.time_till(toc) * 1e-3

    tic.record()
    for C in reversed(range(1, len(surfSrc.tree))):
        PC = surfSrc.tree[C].parent
        upwardSweep(surfSrc.tree, C, PC, param.P, ind0.II, ind0.JJ, ind0.KK,
                    ind0.index, ind0.combII, ind0.combJJ, ind0.combKK,
                    ind0.IImii, ind0.JJmjj, ind0.KKmkk, ind0.index_small,
                    ind0.index_ptr)
    toc.record()
    toc.synchronize()
    timing.time_M2M += tic.time_till(toc) * 1e-3

    tic.record()
    X_Kt = X_Kt[surfSrc.sortSource]
    X_Ktc = X_Ktc[surfSrc.sortSource]
    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc) * 1e-3

    param.Nround = len(surfTar.twig) * param.NCRIT
    Ktx_aux = numpy.zeros(param.Nround)
    Kty_aux = numpy.zeros(param.Nround)
    Ktz_aux = numpy.zeros(param.Nround)

    ### CPU code
    if param.GPU == 0:
        if surfTar.offsetMlt[self, len(surfTar.twig)] > 0:
            Ktx_aux, Kty_aux, Ktz_aux = M2PKt_sort(
                surfSrc, surfTar, Ktx_aux, Kty_aux, Ktz_aux, self,
                ind0.index_large, param, LorY, timing)

        Ktx_aux, Kty_aux, Ktz_aux = P2PKt_sort(surfSrc, surfTar, X_Kt, X_Ktc,
                                               Ktx_aux, Kty_aux, Ktz_aux, self,
                                               LorY, w, param, timing)

    ### GPU code
    elif param.GPU == 1:
        Ktx_gpu = cuda.to_device(Ktx_aux.astype(REAL))
        Kty_gpu = cuda.to_device(Kty_aux.astype(REAL))
        Ktz_gpu = cuda.to_device(Ktz_aux.astype(REAL))

        if surfTar.offsetMlt[self, len(surfTar.twig)] > 0:
            Ktx_gpu, Kty_gpu, Ktz_gpu = M2PKt_gpu(surfSrc, surfTar, Ktx_gpu,
                                                  Kty_gpu, Ktz_gpu, self, ind0,
                                                  param, LorY, timing, kernel)

        Ktx_gpu, Kty_gpu, Ktz_gpu = P2PKt_gpu(surfSrc, surfTar, X_Kt, X_Ktc,
                                              Ktx_gpu, Kty_gpu, Ktz_gpu, self,
                                              LorY, w, param, timing, kernel)

        tic.record()
        Ktx_aux = cuda.from_device(Ktx_gpu, len(Ktx_aux), dtype=REAL)
        Kty_aux = cuda.from_device(Kty_gpu, len(Kty_aux), dtype=REAL)
        Ktz_aux = cuda.from_device(Ktz_gpu, len(Ktz_aux), dtype=REAL)
        toc.record()
        toc.synchronize()
        timing.time_trans += tic.time_till(toc) * 1e-3

    tic.record()
    Kt_lyr = (Ktx_aux[surfTar.unsort]*surfTar.normal[:,0] +
              Kty_aux[surfTar.unsort]*surfTar.normal[:,1] +
              Ktz_aux[surfTar.unsort]*surfTar.normal[:,2])

    if abs(Kt_diag) > 1e-12:  # if same surface
        Kt_lyr += Kt_diag * XKt

    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc) * 1e-3

    return Kt_lyr


def get_phir(XK, XV, surface, xq, Cells, par_reac, ind_reac):
    """
    It computes the reaction potential.
    To compute this potential we need more terms in the Taylor expansion, that
    is the reason why we need fine parameters (par_reac class) and a different
    array of indices (ind_reac) than ind0.

    Arguments
    ----------
    XK      : array, input for the double layer potential.
    XV      : array, input for the single layer potential.
    surface : class, surface where we are computing the reaction potential.
    xq      : array, it contains the position of the charges.
    Cells   : array, it contains the tree cells.
    par_reac: class, fine parameters related to the surface.
    ind_reac: array, it contains the indices related to the treecode
                     computation.

    Returns
    --------
    phi_reac: array, reaction potential.
    AI_int  : int, counter of the amount of near singular integrals solved.
    """

    N = len(XK)
    AI_int = 0

    # Setup vector
    K = par_reac.K
    tic = time.time()
    w = getWeights(K)
    X_V = numpy.zeros(N * K)
    X_Kx = numpy.zeros(N * K)
    X_Ky = numpy.zeros(N * K)
    X_Kz = numpy.zeros(N * K)
    X_Kc = numpy.zeros(N * K)
    X_Vc = numpy.zeros(N * K)

    for i in range(N * K):
        X_V[i] = XV[i // K] * w[i % K] * surface.area[i // K]
        X_Kx[i] = XK[i // K] * w[i % K] * surface.area[
            i // K] * surface.normal[i // K, 0]
        X_Ky[i] = XK[i // K] * w[i % K] * surface.area[
            i // K] * surface.normal[i // K, 1]
        X_Kz[i] = XK[i // K] * w[i % K] * surface.area[
            i // K] * surface.normal[i // K, 2]
        X_Kc[i] = XK[i // K]
        X_Vc[i] = XV[i // K]

    toc = time.time()

    # P2M
    tic = time.time()
    C = 0
    getMultipole(Cells, C, surface.xj, surface.yj, surface.zj, X_V, X_Kx, X_Ky,
                 X_Kz, ind_reac, par_reac.P, par_reac.NCRIT)
    toc = time.time()
    time_P2M = toc - tic

    # M2M
    tic = time.time()
    for C in reversed(range(1, len(Cells))):
        PC = Cells[C].parent
        upwardSweep(Cells, C, PC, par_reac.P, ind_reac.II, ind_reac.JJ,
                    ind_reac.KK, ind_reac.index, ind_reac.combII,
                    ind_reac.combJJ, ind_reac.combKK, ind_reac.IImii,
                    ind_reac.JJmjj, ind_reac.KKmkk, ind_reac.index_small,
                    ind_reac.index_ptr)
    toc = time.time()
    time_M2M = toc - tic

    # Evaluation
    IorE = 0  # This evaluation is on charge points, no self-operator
    # 0 means it doesn't matter if it is internal or external.
    AI_int = 0
    phi_reac = numpy.zeros(len(xq))
    time_P2P = 0.
    time_M2P = 0.
    for i in range(len(xq)):
        CJ = 0
        Kval = 0.
        Vval = 0.
        source = []
        Kval, Vval, source, time_M2P = M2P_nonvec(Cells, CJ, xq[i], Kval, Vval,
                                                  ind_reac.index_large,
                                                  par_reac, source, time_M2P)
        Kval, Vval, AI_int, time_P2P = P2P_nonvec(
            Cells, surface, X_V, X_Kx, X_Ky, X_Kz, X_Kc, X_Vc, xq[i], Kval,
            Vval, IorE, par_reac, w, source, AI_int, time_P2P)
        phi_reac[i] = (-Kval + Vval) / (4 * pi)

    return phi_reac, AI_int


def get_phir_gpu(XK, XV, surface, field, par_reac, kernel):
    """
    It computes the reaction potential on the GPU  and it brings the data
    to the cpu.

    Arguments
    ----------
    XK      : array, input for the double layer potential.
    XV      : array, input for the single layer potential.
    surface : class, surface where we are computing the reaction potential.
    field   : class, information about the different regions in the molecule.
    par_reac: class, fine parameters related to the surface.

    Returns
    --------
    phir_cpu: array, reaction potential brought from the GPU to the cpu.
    AI_int  : int, counter of the amount of near singular integrals solved.
    """

    REAL = par_reac.REAL
    Nq = len(field.xq)
    N = len(XK)
    AI_int = 0

    # Setup vector
    K = par_reac.K
    tic = time.time()
    w = getWeights(K)
    X_V = numpy.zeros(N * K)
    X_Kx = numpy.zeros(N * K)
    X_Ky = numpy.zeros(N * K)
    X_Kz = numpy.zeros(N * K)
    X_Kc = numpy.zeros(N * K)
    X_Vc = numpy.zeros(N * K)

    for i in range(N * K):
        X_V[i] = XV[i // K] * w[i % K] * surface.area[i // K]
        X_Kx[i] = XK[i // K] * w[i % K] * surface.area[
            i // K] * surface.normal[i // K, 0]
        X_Ky[i] = XK[i // K] * w[i % K] * surface.area[
            i // K] * surface.normal[i // K, 1]
        X_Kz[i] = XK[i // K] * w[i % K] * surface.area[
            i // K] * surface.normal[i // K, 2]
        X_Kc[i] = XK[i // K]
        X_Vc[i] = XV[i // K]

    toc = time.time()
    sort = surface.sortSource
    phir = cuda.to_device(numpy.zeros(Nq, dtype=REAL))
    m_gpu = cuda.to_device(X_V[sort].astype(REAL))
    mx_gpu = cuda.to_device(X_Kx[sort].astype(REAL))
    my_gpu = cuda.to_device(X_Ky[sort].astype(REAL))
    mz_gpu = cuda.to_device(X_Kz[sort].astype(REAL))
    mKc_gpu = cuda.to_device(X_Kc[sort].astype(REAL))
    mVc_gpu = cuda.to_device(X_Vc[sort].astype(REAL))
    AI_int_gpu = cuda.to_device(numpy.zeros(Nq, dtype=numpy.int32))
    xkDev = cuda.to_device(surface.xk.astype(REAL))
    wkDev = cuda.to_device(surface.wk.astype(REAL))

    get_phir = kernel.get_function("get_phir")
    GSZ = int(numpy.ceil(float(Nq) / par_reac.BSZ))

    get_phir(phir,
             field.xq_gpu,
             field.yq_gpu,
             field.zq_gpu,
             m_gpu,
             mx_gpu,
             my_gpu,
             mz_gpu,
             mKc_gpu,
             mVc_gpu,
             surface.xjDev,
             surface.yjDev,
             surface.zjDev,
             surface.AreaDev,
             surface.kDev,
             surface.vertexDev,
             numpy.int32(len(surface.xj)),
             numpy.int32(Nq),
             numpy.int32(par_reac.K),
             xkDev,
             wkDev,
             REAL(par_reac.threshold),
             AI_int_gpu,
             numpy.int32(len(surface.xk)),
             surface.XskDev,
             surface.WskDev,
             block=(par_reac.BSZ, 1, 1),
             grid=(GSZ, 1))

    AI_aux = numpy.zeros(Nq, dtype=numpy.int32)
    AI_aux = cuda.from_device(AI_int_gpu, Nq, dtype=numpy.int32)
    AI_int = numpy.sum(AI_aux)

    phir_cpu = numpy.zeros(Nq, dtype=REAL)
    phir_cpu = cuda.from_device(phir, Nq, dtype=REAL)

    return phir_cpu, AI_int
