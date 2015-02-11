'''
  Copyright (C) 2013 by Christopher Cooper, Lorena Barba

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
'''

import sys 
sys.path.append('tree')
from FMMutils import *
import pycuda.autoinit
import pycuda.driver as cuda
import time


def getWeights(K):

    w = zeros(K)
    if K==1:
        w[0] = 1
    if K==3:
        w[0] = 1/3.
        w[1] = 1/3.
        w[2] = 1/3.
    if K==4:
        w[0] = -27./48
        w[1] =  25./48
        w[2] =  25./48
        w[3] =  25./48
    if K==7:
        w[0] = 0.225
        w[1] = 0.125939180544827
        w[2] = 0.125939180544827
        w[3] = 0.125939180544827
        w[4] = 0.132394152788506
        w[5] = 0.132394152788506
        w[6] = 0.132394152788506

    return w

def project(XK, XV, LorY, surfSrc, surfTar, K_diag, V_diag, IorE,
            self, param, ind0, timing, kernel):

    tic = cuda.Event()
    toc = cuda.Event()

    REAL = param.REAL
    Ns = len(surfSrc.triangle)
    Nt = len(surfTar.triangle)
    L = sqrt(2*surfSrc.Area) # Representative length

    tic.record()
    K = param.K
    w    = getWeights(K)
    X_V  = zeros(Ns*K)
    X_Kx = zeros(Ns*K)
    X_Ky = zeros(Ns*K)
    X_Kz = zeros(Ns*K)
    X_Kc = zeros(Ns*K)
    X_Vc = zeros(Ns*K)

    NsK = arange(Ns*K)
    X_V[:]  =  XV[NsK/K]*w[NsK%K]*surfSrc.Area[NsK/K]
    X_Kx[:] = XK[NsK/K]*w[NsK%K]*surfSrc.Area[NsK/K]*surfSrc.normal[NsK/K,0]
    X_Ky[:] = XK[NsK/K]*w[NsK%K]*surfSrc.Area[NsK/K]*surfSrc.normal[NsK/K,1]
    X_Kz[:] = XK[NsK/K]*w[NsK%K]*surfSrc.Area[NsK/K]*surfSrc.normal[NsK/K,2]
    X_Kc[:] = XK[NsK/K]
    X_Vc[:] = XV[NsK/K]

    toc.record()
    toc.synchronize()
    timing.time_mass += tic.time_till(toc)*1e-3

    tic.record()
    C = 0
    getMultipole(surfSrc.tree, C, surfSrc.xj, surfSrc.yj, surfSrc.zj, 
                    X_V, X_Kx, X_Ky, X_Kz, ind0, param.P, param.NCRIT)
    toc.record()
    toc.synchronize()
    timing.time_P2M += tic.time_till(toc)*1e-3


    tic.record()
    for C in reversed(range(1,len(surfSrc.tree))):
        PC = surfSrc.tree[C].parent
        upwardSweep(surfSrc.tree, C, PC, param.P, ind0.II, ind0.JJ, ind0.KK, ind0.index, ind0.combII, ind0.combJJ, 
                    ind0.combKK, ind0.IImii, ind0.JJmjj, ind0.KKmkk, ind0.index_small, ind0.index_ptr)
    toc.record()
    toc.synchronize()
    timing.time_M2M += tic.time_till(toc)*1e-3

    tic.record()
    X_V = X_V[surfSrc.sortSource]
    X_Kx = X_Kx[surfSrc.sortSource]
    X_Ky = X_Ky[surfSrc.sortSource]
    X_Kz = X_Kz[surfSrc.sortSource]
    X_Kc = X_Kc[surfSrc.sortSource]
    X_Vc = X_Vc[surfSrc.sortSource]
    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc)*1e-3

    param.Nround = len(surfTar.twig)*param.NCRIT
    K_aux  = zeros(param.Nround)
    V_aux  = zeros(param.Nround)
    AI_int = 0

    ### CPU code
    if param.GPU==0:
        K_aux, V_aux = M2P_sort(surfSrc, surfTar, K_aux, V_aux, self, 
                                ind0.index_large, param, LorY, timing)

        K_aux, V_aux = P2P_sort(surfSrc, surfTar, X_V, X_Kx, X_Ky, X_Kz, X_Kc, X_Vc, 
                                K_aux, V_aux, self, LorY, K_diag, V_diag, IorE, L, w, param, timing)

    ### GPU code
    elif param.GPU==1:
        K_gpu = cuda.to_device(K_aux.astype(REAL))
        V_gpu = cuda.to_device(V_aux.astype(REAL))

        if surfTar.offsetMlt[self,len(surfTar.twig)]>0:
            K_gpu, V_gpu = M2P_gpu(surfSrc, surfTar, K_gpu, V_gpu, self, 
                                    ind0, param, LorY, timing, kernel)

        K_gpu, V_gpu = P2P_gpu(surfSrc, surfTar, X_V, X_Kx, X_Ky, X_Kz, X_Kc, X_Vc, 
                                K_gpu, V_gpu, self, LorY, K_diag, IorE, L, w, param, timing, kernel)

        tic.record()
        K_aux = cuda.from_device(K_gpu, len(K_aux), dtype=REAL)
        V_aux = cuda.from_device(V_gpu, len(V_aux), dtype=REAL)
        toc.record()
        toc.synchronize()
        timing.time_trans += tic.time_till(toc)*1e-3

    tic.record()
    K_lyr = K_aux[surfTar.unsort]
    V_lyr = V_aux[surfTar.unsort]
    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc)*1e-3

    return K_lyr, V_lyr 
    
def project_Kt(XKt, LorY, surfSrc, surfTar, Kt_diag,
                self, param, ind0, timing, kernel):

    tic = cuda.Event()
    toc = cuda.Event()

    REAL = param.REAL
    Ns = len(surfSrc.triangle)
    Nt = len(surfTar.triangle)
    L = sqrt(2*surfSrc.Area) # Representative length

    tic.record()
    K = param.K
    w    = getWeights(K)
    X_Kt = zeros(Ns*K)
    X_Ktc = zeros(Ns*K)

    NsK = arange(Ns*K)
    X_Kt[:]  = XKt[NsK/K]*w[NsK%K]*surfSrc.Area[NsK/K]
    X_Ktc[:] = XKt[NsK/K]

    toc.record()
    toc.synchronize()
    timing.time_mass += tic.time_till(toc)*1e-3

    tic.record()
    C = 0
    X_aux = zeros(Ns*K)
    getMultipole(surfSrc.tree, C, surfSrc.xj, surfSrc.yj, surfSrc.zj, 
                    X_Kt, X_aux, X_aux, X_aux, ind0, param.P, param.NCRIT)
    toc.record()
    toc.synchronize()
    timing.time_P2M += tic.time_till(toc)*1e-3


    tic.record()
    for C in reversed(range(1,len(surfSrc.tree))):
        PC = surfSrc.tree[C].parent
        upwardSweep(surfSrc.tree, C, PC, param.P, ind0.II, ind0.JJ, ind0.KK, ind0.index, ind0.combII, ind0.combJJ, 
                    ind0.combKK, ind0.IImii, ind0.JJmjj, ind0.KKmkk, ind0.index_small, ind0.index_ptr)
    toc.record()
    toc.synchronize()
    timing.time_M2M += tic.time_till(toc)*1e-3

    tic.record()
    X_Kt = X_Kt[surfSrc.sortSource]
    X_Ktc = X_Ktc[surfSrc.sortSource]
    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc)*1e-3

    param.Nround = len(surfTar.twig)*param.NCRIT
    Ktx_aux  = zeros(param.Nround)
    Kty_aux  = zeros(param.Nround)
    Ktz_aux  = zeros(param.Nround)
    AI_int = 0

    ### CPU code
    if param.GPU==0:
        if surfTar.offsetMlt[self,len(surfTar.twig)]>0:
            Ktx_aux, Kty_aux, Ktz_aux = M2PKt_sort(surfSrc, surfTar, Ktx_aux, Kty_aux, Ktz_aux, self, 
                                    ind0.index_large, param, LorY, timing)

        Ktx_aux, Kty_aux, Ktz_aux = P2PKt_sort(surfSrc, surfTar, X_Kt, X_Ktc, 
                            Ktx_aux, Kty_aux, Ktz_aux, self, LorY, w, param, timing)

    ### GPU code
    elif param.GPU==1:
        Ktx_gpu = cuda.to_device(Ktx_aux.astype(REAL))
        Kty_gpu = cuda.to_device(Kty_aux.astype(REAL))
        Ktz_gpu = cuda.to_device(Ktz_aux.astype(REAL))

        if surfTar.offsetMlt[self,len(surfTar.twig)]>0:
            Ktx_gpu, Kty_gpu, Ktz_gpu = M2PKt_gpu(surfSrc, surfTar, 
                                    Ktx_gpu, Kty_gpu, Ktz_gpu, self, 
                                    ind0, param, LorY, timing, kernel)

        Ktx_gpu, Kty_gpu, Ktz_gpu = P2PKt_gpu(surfSrc, surfTar, X_Kt, X_Ktc, Ktx_gpu, Kty_gpu, Ktz_gpu, 
                                self, LorY, w, param, timing, kernel)

        tic.record()
        Ktx_aux = cuda.from_device(Ktx_gpu, len(Ktx_aux), dtype=REAL)
        Kty_aux = cuda.from_device(Kty_gpu, len(Kty_aux), dtype=REAL)
        Ktz_aux = cuda.from_device(Ktz_gpu, len(Ktz_aux), dtype=REAL)
        toc.record()
        toc.synchronize()
        timing.time_trans += tic.time_till(toc)*1e-3

    tic.record()
    Kt_lyr = Ktx_aux[surfTar.unsort]*surfTar.normal[:,0] \
           + Kty_aux[surfTar.unsort]*surfTar.normal[:,1] \
           + Ktz_aux[surfTar.unsort]*surfTar.normal[:,2] 

    if abs(Kt_diag)>1e-12: # if same surface
        Kt_lyr += Kt_diag * XKt

    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc)*1e-3

    return Kt_lyr


def project_Ktqual(XKt, LorY, surfSrc, surfTar, Kt_diag,
                self, param, ind0, timing, kernel):

    tic = cuda.Event()
    toc = cuda.Event()

    REAL = param.REAL
    Ns = len(surfSrc.triangle)
    Nt = len(surfTar.triangle)

    tic.record()
    K = param.K
    w = getWeights(K)

    X_Kt  = XKt*surfSrc.Area
    X_Ktc = X_Kt/w[0]       # k=0 will store the fine integration

    toc.record()
    toc.synchronize()
    timing.time_mass += tic.time_till(toc)*1e-3

    tic.record()
    C = 0
    X_aux = zeros(Ns)

    getMultipole(surfSrc.tree, C, surfSrc.xj, surfSrc.yj, surfSrc.zj, 
                    X_Kt, X_aux, X_aux, X_aux, ind0, param.P, param.NCRIT)
    toc.record()
    toc.synchronize()
    timing.time_P2M += tic.time_till(toc)*1e-3


    tic.record()
    for C in reversed(range(1,len(surfSrc.tree))):
        PC = surfSrc.tree[C].parent
        upwardSweep(surfSrc.tree, C, PC, param.P, ind0.II, ind0.JJ, ind0.KK, ind0.index, ind0.combII, ind0.combJJ, 
                    ind0.combKK, ind0.IImii, ind0.JJmjj, ind0.KKmkk, ind0.index_small, ind0.index_ptr)
    toc.record()
    toc.synchronize()
    timing.time_M2M += tic.time_till(toc)*1e-3

    tic.record()
    X_Kt  = X_Kt[surfSrc.sortSource]
    X_Ktc = X_Ktc[surfSrc.sortSource]
    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc)*1e-3

    param.Nround = len(surfTar.twig)*param.NCRIT*param.K

    Ktx_aux  = zeros(param.Nround)
    Kty_aux  = zeros(param.Nround)
    Ktz_aux  = zeros(param.Nround)
    AI_int = 0

    ### CPU code
    if param.GPU==0:
        if surfTar.offsetMlt[self,len(surfTar.twig)]>0:
            Ktx_aux, Kty_aux, Ktz_aux = M2PKt_sort(surfSrc, surfTar, Ktx_aux, Kty_aux, Ktz_aux, self, 
                                    ind0.index_large, param, LorY, timing)

        Ktx_aux, Kty_aux, Ktz_aux = P2PKtqual_sort(surfSrc, surfTar, X_Kt, X_Ktc,
                            Ktx_aux, Kty_aux, Ktz_aux, self, LorY, w, param, timing)

    ### GPU code
    elif param.GPU==1:
        Ktx_gpu = cuda.to_device(Ktx_aux.astype(REAL))
        Kty_gpu = cuda.to_device(Kty_aux.astype(REAL))
        Ktz_gpu = cuda.to_device(Ktz_aux.astype(REAL))

        if surfTar.offsetMlt[self,len(surfTar.twig)]>0:
            Ktx_gpu, Kty_gpu, Ktz_gpu = M2PKt_gpu(surfSrc, surfTar, 
                                    Ktx_gpu, Kty_gpu, Ktz_gpu, self, 
                                    ind0, param, LorY, timing, kernel)

        Ktx_gpu, Kty_gpu, Ktz_gpu = P2PKt_gpu(surfSrc, surfTar, X_Kt, X_Ktc, Ktx_gpu, Kty_gpu, Ktz_gpu, 
                                self, LorY, w, param, timing, kernel)

        tic.record()
        Ktx_aux = cuda.from_device(Ktx_gpu, len(Ktx_aux), dtype=REAL)
        Kty_aux = cuda.from_device(Kty_gpu, len(Kty_aux), dtype=REAL)
        Ktz_aux = cuda.from_device(Ktz_gpu, len(Ktz_aux), dtype=REAL)
        toc.record()
        toc.synchronize()
        timing.time_trans += tic.time_till(toc)*1e-3

    tic.record()

    Ktx_aux = Ktx_aux[surfTar.unsort]
    Kty_aux = Kty_aux[surfTar.unsort]
    Ktz_aux = Ktz_aux[surfTar.unsort]

    Ktx = zeros(Nt)
    Kty = zeros(Nt)
    Ktz = zeros(Nt)

    for i in range(Nt):
#        indices = int32(linspace(i, Nt*(param.K-1)+i, num=param.K))
        indices = arange(param.K*i,param.K*i+param.K)
        Ktx[i] = sum(Ktx_aux[indices]*w)
        Kty[i] = sum(Kty_aux[indices]*w)
        Ktz[i] = sum(Ktz_aux[indices]*w)

    Kt_lyr = surfTar.Area * (Ktx*surfTar.normal[:,0] \
                           + Kty*surfTar.normal[:,1] \
                           + Ktz*surfTar.normal[:,2])

    if abs(Kt_diag.any())>1e-12: # if same surface
        Kt_lyr += Kt_diag * XKt
#    if sum(abs(Ktx))>1e-10:
#        savetxt('test1',Kt_lyr)
#        quit()

    toc.record()
    toc.synchronize()
    timing.time_sort += tic.time_till(toc)*1e-3

    return Kt_lyr


def get_phir (XK, XV, surface, xq, Cells, par_reac, ind_reac):

    REAL = par_reac.REAL
    N = len(XK)
    MV = zeros(len(XK))
    L = sqrt(2*surface.Area) # Representative length
    AI_int = 0

    # Setup vector
    K = par_reac.K
    tic = time.time()
    w    = getWeights(K)
    X_V = zeros(N*K)
    X_Kx = zeros(N*K)
    X_Ky = zeros(N*K)
    X_Kz = zeros(N*K)
    X_Kc = zeros(N*K)
    X_Vc = zeros(N*K)

    for i in range(N*K):
        X_V[i]   = XV[i/K]*w[i%K]*surface.Area[i/K]
        X_Kx[i]  = XK[i/K]*w[i%K]*surface.Area[i/K]*surface.normal[i/K,0]
        X_Ky[i]  = XK[i/K]*w[i%K]*surface.Area[i/K]*surface.normal[i/K,1]
        X_Kz[i]  = XK[i/K]*w[i%K]*surface.Area[i/K]*surface.normal[i/K,2]
        X_Kc[i]  = XK[i/K]
        X_Vc[i]  = XV[i/K]
    
    toc = time.time()
    time_set = toc - tic

    # P2M
    tic = time.time()
    C = 0
    getMultipole(Cells, C, surface.xj, surface.yj, surface.zj, 
                X_V, X_Kx, X_Ky, X_Kz, ind_reac, par_reac.P, par_reac.NCRIT)
    toc = time.time()
    time_P2M = toc - tic

    # M2M
    tic = time.time()
    for C in reversed(range(1,len(Cells))):
        PC = Cells[C].parent
        upwardSweep(Cells, C, PC, par_reac.P, ind_reac.II, ind_reac.JJ, ind_reac.KK, ind_reac.index, 
                    ind_reac.combII, ind_reac.combJJ, ind_reac.combKK, ind_reac.IImii, ind_reac.JJmjj, 
                    ind_reac.KKmkk, ind_reac.index_small, ind_reac.index_ptr)
    toc = time.time()
    time_M2M = toc - tic

    # Evaluation
    IorE = 0    # This evaluation is on charge points, no self-operator
    AI_int = 0
    phi_reac = zeros(len(xq))
    time_P2P = 0.
    time_M2P = 0.
    for i in range(len(xq)):
        CJ = 0
        Kval = 0.
        Vval = 0.
        source = []
        Kval, Vval, source, time_M2P = M2P_nonvec(Cells, CJ, xq[i], Kval, Vval,
                                                 ind_reac.index_large, par_reac, source, time_M2P)
        Kval, Vval, AI_int, time_P2P = P2P_nonvec(Cells, surface, X_V, X_Kx, X_Ky, X_Kz, X_Kc, X_Vc,
                                        xq[i], Kval, Vval, IorE, par_reac, w, source, AI_int, time_P2P)
        phi_reac[i] = (-Kval + Vval)/(4*pi)
#    print '\tTime set: %f'%time_P2M
#    print '\tTime P2M: %f'%time_P2M
#    print '\tTime M2M: %f'%time_M2M
#    print '\tTime M2P: %f'%time_M2P
#    print '\tTime P2P: %f'%time_P2P

    return phi_reac, AI_int

def get_phir_gpu (XK, XV, surface, field, par_reac, kernel):

    REAL = par_reac.REAL
    Nq = len(field.xq)
    N = len(XK)
    MV = zeros(len(XK))
    L = sqrt(2*surface.Area) # Representative length
    AI_int = 0

    # Setup vector
    K = par_reac.K
    tic = time.time()
    w    = getWeights(K)
    X_V = zeros(N*K)
    X_Kx = zeros(N*K)
    X_Ky = zeros(N*K)
    X_Kz = zeros(N*K)
    X_Kc = zeros(N*K)
    X_Vc = zeros(N*K)

    for i in range(N*K):
        X_V[i]   = XV[i/K]*w[i%K]*surface.Area[i/K]
        X_Kx[i]  = XK[i/K]*w[i%K]*surface.Area[i/K]*surface.normal[i/K,0]
        X_Ky[i]  = XK[i/K]*w[i%K]*surface.Area[i/K]*surface.normal[i/K,1]
        X_Kz[i]  = XK[i/K]*w[i%K]*surface.Area[i/K]*surface.normal[i/K,2]
        X_Kc[i]  = XK[i/K]
        X_Vc[i]  = XV[i/K]
    
    toc = time.time()
    time_set = toc - tic
    
    sort = surface.sortSource
    phir = cuda.to_device(zeros(Nq, dtype=REAL))
    m_gpu   = cuda.to_device(X_V[sort].astype(REAL))
    mx_gpu  = cuda.to_device(X_Kx[sort].astype(REAL))
    my_gpu  = cuda.to_device(X_Ky[sort].astype(REAL))
    mz_gpu  = cuda.to_device(X_Kz[sort].astype(REAL))
    mKc_gpu = cuda.to_device(X_Kc[sort].astype(REAL))
    mVc_gpu = cuda.to_device(X_Vc[sort].astype(REAL))
    AI_int_gpu = cuda.to_device(zeros(Nq, dtype=int32))
    xkDev = cuda.to_device(surface.xk.astype(REAL))
    wkDev = cuda.to_device(surface.wk.astype(REAL))


    get_phir = kernel.get_function("get_phir")
    
    GSZ = int(ceil(float(Nq)/par_reac.BSZ))

    get_phir(phir, field.xq_gpu, field.yq_gpu, field.zq_gpu, m_gpu, mx_gpu, my_gpu, mz_gpu, mKc_gpu, mVc_gpu, 
            surface.xjDev, surface.yjDev, surface.zjDev, surface.AreaDev, surface.kDev, surface.vertexDev, 
            int32(len(surface.xj)), int32(Nq), int32(par_reac.K), xkDev, wkDev, REAL(par_reac.threshold),
            AI_int_gpu, int32(len(surface.xk)), surface.XskDev, surface.WskDev, block=(par_reac.BSZ,1,1), grid=(GSZ,1))

    AI_aux = zeros(Nq, dtype=int32)
    AI_aux = cuda.from_device(AI_int_gpu, Nq, dtype=int32)
    AI_int = sum(AI_aux)

    phir_cpu = zeros(Nq, dtype=REAL)
    phir_cpu = cuda.from_device(phir, Nq, dtype=REAL)

    return phir_cpu, AI_int
