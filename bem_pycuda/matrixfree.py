'''
    Copyright (C) 2011 by Christopher Cooper, Lorena Barba
  
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
from numpy    import float64 as REAL
import pycuda.gpuarray  as gpuarray
import pycuda.driver    as cuda
from cuda_kernels import kernels
import time

def getGaussPoints(y,triangle, n): 
    # y         : vertices
    # triangle  : array with indices for corresponding triangles
    # n         : Gauss points per element

    N  = len(triangle) # Number of triangles
    xi = zeros((N*n,3))
    if n==1:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[i,:] = dot(M, 1/3.*ones(3))

    if n==3:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[n*i,:] = dot(M, array([0.5, 0.5, 0.]))
            xi[n*i+1,:] = dot(M, array([0., 0.5, 0.5]))
            xi[n*i+2,:] = dot(M, array([0.5, 0., 0.5]))

    if n==4:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[n*i,:] = dot(M, array([1/3., 1/3., 1/3.]))
            xi[n*i+1,:] = dot(M, array([3/5., 1/5., 1/5.]))
            xi[n*i+2,:] = dot(M, array([1/5., 3/5., 1/5.]))
            xi[n*i+3,:] = dot(M, array([1/5., 1/5., 3/5.]))

    if n==7:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[n*i+0,:] = dot(M, array([1/3.,1/3.,1/3.]))
            xi[n*i+1,:] = dot(M, array([.79742699,.10128651,.10128651]))
            xi[n*i+2,:] = dot(M, array([.10128651,.79742699,.10128651]))
            xi[n*i+3,:] = dot(M, array([.10128651,.10128651,.79742699]))
            xi[n*i+4,:] = dot(M, array([.05971587,.47014206,.47014206]))
            xi[n*i+5,:] = dot(M, array([.47014206,.05971587,.47014206]))
            xi[n*i+6,:] = dot(M, array([.47014206,.47014206,.05971587]))

    return xi[:,0], xi[:,1], xi[:,2]

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
        w[1] = 0.12593918
        w[2] = 0.12593918
        w[3] = 0.12593918
        w[4] = 0.13239415
        w[5] = 0.13239415
        w[6] = 0.13239415
    return w

def gmres_dot (Precond, E_hat, X, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, 
				AreaHost, AreaDev, normal_xDev, xj, yj, zj, xi, yi, zi, xtHost, ytHost, ztHost, 
				xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, 
				xcDev, ycDev, zcDev, sizeTarDev, offsetMltDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, 
				Pre1Host, Pre2Host, Pre3Host, tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, 
				offsetTarHost, sizeTarHost,  offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, Cells, theta, Nm, II, JJ, KK, 
                index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, P, kappa, NCRIT, K, threshold, BSZ, GSZ, 
				BlocksPerTwig, twig, eps, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack):
    # Precond           : preconditioner
    # E_hat             : coefficient of bottom right block
    # X                 : vector of weights
    # vertex            : array of position of vertices
    # triangle          : array of indices of triangle vertices in vertex array
    # triHost/Dev       : packed array of triangle indices corresponding to each Gauss point on host/device memory  
    # kHost/Dev         : packed array of Gauss points indices within the element corresponding to each Gauss point on host/device memory   
    # vertexHost/Dev    : packed array of vertices of triangles on host/device memory
    # AreaHost/Dev      : packed array of area of triangles on host/device memory
    # normal_xDev       : packed array of x component of normal vector to triangles on device memory
    # xi, yi, zi        : position of targets
    # xj, yj, zj        : position of sources
    # xt,yt,ztHost/Dev  : packed array with position of targets (collocation points) on host/device memory
    # xs,ys,zsHost/Dev  : packed array with position of sources (Gauss points) on host/device memory
    # xc,yc,zcHost/Dev  : packed array with position of box centers on host/device memory
    # sizeTarHost/Dev   : array with number of targets per twig cell on host/device memory
    # offsetMltHost/Dev : array with pointers to first element of each twig in xcHost/Dev array on host/device memory
    # Pre0,1,2,3Host/Dev: packed array with diagonal values of preconditioner for blocks 0,1,2,3 on host/device
    # tarPtr            : packed array with pointers to targets in xi, yi, zi array
    # srcPtr            : packed array with pointers to sources in xj, yj, zj array
    # offSrc            : length of array of packed sources
    # mltPtr            : packed array with pointers to multipoles in Cells array
    # offMlt            : length of array of packed multipoles
    # offsetSrcHost/Dev : array with pointers to first element of each twig in xsHost/Dev array on host/device memory
    # Area              : array of element area 
    # normal            : array of elements normal
    # xk/Dev, wk/Dev    : position and weight of 1D gauss quadrature on host/device memory
    # Cells             : array of Cells
    # K                 : number of 2D Gauss points per element
    # threshold         : threshold to change from analytical to Gauss integration
    # BSZ, GSZ          : block size and grid size for CUDA
    # blocksPerTwig     : number of CUDA blocks that fit on a twig (NCRIT/BSZ)
    # b                 : RHS vector
    # R                 : number of iterations to restart
    # tol               : GMRES tolerance
    # max_iter          : maximum number of iterations
    # theta             : MAC criterion
    # Nm                : number of terms in Taylor expansion
    # II, JJ, KK        : x,y,z powers of multipole expansion
    # index             : 1D mapping of II,JJ,KK (index of multipoles)
    # P                 : order of expansion
    # kappa             : reciprocal of Debye length
    # NCRIT             : max number of points per twig cell
    # twig              : array of indices of twigs in Cells array


    N = len(triangle)
    MV = zeros(len(X))
    L = sqrt(2*Area) # Representative length of panel
    AI_int = 0
    
	### Set up weights vector for Treecode
	### so that matrix has only 1/r type elements (with no coefficients)
    w    = getWeights(K)
    X_K = zeros(N*K)
    X_Kx = zeros(N*K)
    X_Ky = zeros(N*K)
    X_Kz = zeros(N*K)
    
    for i in range(N*K):
        X_K[i] = X[i/K+N]*w[i%K]*Area[i/K]
        X_Kx[i]  = X[i/K]*w[i%K]*Area[i/K]*normal[i/K,0]
        X_Ky[i]  = X[i/K]*w[i%K]*Area[i/K]*normal[i/K,1]
        X_Kz[i]  = X[i/K]*w[i%K]*Area[i/K]*normal[i/K,2]

	### P2M
    tic = time.time()
    C = 0
    getMultipole(Cells, C, xj, yj, zj, X_K, X_Kx, X_Ky, X_Kz, II, JJ, KK, index, P, NCRIT)
    toc = time.time()
    time_P2M += toc - tic

	### M2M
    tic = time.time()
    for C in reversed(range(1,len(Cells))):
        PC = Cells[C].parent
        upwardSweep(Cells,C,PC,P, II, JJ, KK, index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small)
    toc = time.time()
    time_M2M += toc - tic

	### Pack weights and multipoles
    tic = time.time()
	# Weights
    mHost = X_K[srcPtr[0:offSrc]]
    mxHost = X_Kx[srcPtr[0:offSrc]]
    myHost = X_Ky[srcPtr[0:offSrc]]
    mzHost = X_Kz[srcPtr[0:offSrc]]

	# Multipoles
    mpHost = zeros(offMlt*Nm)
    mpxHost = zeros(offMlt*Nm)
    mpyHost = zeros(offMlt*Nm)
    mpzHost = zeros(offMlt*Nm)

    i = -1
    for C in mltPtr[0:offMlt]:
        i += 1
        mpHost[i*Nm:i*Nm+Nm] = Cells[C].M
        mpxHost[i*Nm:i*Nm+Nm] = Cells[C].Mx
        mpyHost[i*Nm:i*Nm+Nm] = Cells[C].My
        mpzHost[i*Nm:i*Nm+Nm] = Cells[C].Mz
    toc = time.time()
    time_pack += toc - tic

	### Transfer packed arrays to GPU
    mpDev = gpuarray.to_gpu(mpHost) 
    mpxDev = gpuarray.to_gpu(mpxHost) 
    mpyDev = gpuarray.to_gpu(mpyHost) 
    mpzDev = gpuarray.to_gpu(mpzHost) 

    mDev = gpuarray.to_gpu(mHost) 
    mxDev = gpuarray.to_gpu(mxHost) 
    myDev = gpuarray.to_gpu(myHost) 
    mzDev = gpuarray.to_gpu(mzHost) 

    tic = time.time()

	### Allocate result vector
    p  = zeros(len(X))
    p1Dev = gpuarray.zeros(len(xtHost), dtype=REAL) # Top half of vector
    p2Dev = gpuarray.zeros(len(xtHost), dtype=REAL) # Bottom half of vector

    xkSize = len(xk)
    mod = kernels(BSZ,Nm,xkSize)

    tec = cuda.Event()
    tac = cuda.Event()
    tec.record()
	# M2P
    M2P_gpu = mod.get_function("M2P")
    if len(mpHost>0):
        M2P_gpu(sizeTarDev, offsetMltDev, xtDev, ytDev, ztDev, xcDev, ycDev, zcDev, 
                mpDev, mpxDev, mpyDev, mpzDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, p1Dev, p2Dev, 
                int32(N), int32(P), REAL(kappa), REAL(E_hat), int32(BlocksPerTwig), int32(NCRIT), 
                block=(BSZ,1,1),grid=(GSZ,1))
    # M2P on CPU
#    p = M2P_pack(tarPtr, xtHost, ytHost, ztHost, offsetTarHost, sizeTarHost, 
#               xcHost, ycHost, zcHost, mpHost, mpxHost, mpyHost, mpzHost,
#               Pre0Host, Pre1Host, Pre2Host, Pre3Host,
#               offsetMltHost, p, P, kappa, Nm, N, Precond, E_hat)
    tac.record()
    tac.synchronize()
    time_M2P += tec.time_till(tac)*1e-3

	# P2P
    tec.record()
    AI_int_gpu = gpuarray.zeros(len(xtHost), dtype=int32)
    P2P_gpu = mod.get_function("P2P")
    P2P_gpu(offsetSrcDev, sizeTarDev, triDev, kDev, 
           xsDev, ysDev, zsDev, mDev, mxDev, myDev, mzDev, xtDev, ytDev, ztDev, 
            AreaDev, p1Dev, p2Dev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, REAL(E_hat), 
            int32(N), vertexDev, normal_xDev, int32(K), REAL(w[0]), xkDev, wkDev, 
            REAL(kappa), REAL(threshold), REAL(eps), int32(BlocksPerTwig), int32(NCRIT), 
            AI_int_gpu, block=(BSZ,1,1), grid=(GSZ,1))

    # P2P on CPU
#    p, AI_int, time_an = P2P_pack(xi, yi, zi, offsetSrcHost, offsetTarHost, sizeTarHost, 
#                                tarPtr, srcPtr, xsHost, ysHost, zsHost, mHost, mxHost, myHost, mzHost, 
#                                xtHost, ytHost, ztHost, E_hat, N, p, vertexHost, vertex, triangle, AreaHost, Area, normal[:,0], 
#                                triHost, kHost, K, w, xk, wk, Precond, kappa, threshold, eps, time_an, AI_int)


    tac.record()
    tac.synchronize()
    time_P2P += tec.time_till(tac)*1e-3

	### Unpack resulting vector
    paux1 = zeros(len(xtHost), dtype=REAL) # Top half of vector
    paux2 = zeros(len(xtHost), dtype=REAL) # Bottom half of vector
    p1Dev.get(paux1)
    p2Dev.get(paux2)
    p_test = zeros(len(p))
    AI_aux = zeros(len(xtHost), dtype=int32)
    AI_int_gpu.get(AI_aux)
    AI_int_cpu = zeros(len(p))
    t = -1
    for CI in twig:
        t += 1
        CI_start = offsetTarHost[t]
        CI_end = offsetTarHost[t] + sizeTarHost[t]
        targets = tarPtr[CI_start:CI_end]
        AI_int_cpu[targets] += AI_aux[CI_start:CI_end]
        p[targets]   += paux1[CI_start:CI_end]
        p[targets+N] += paux2[CI_start:CI_end]

    AI_int = sum(AI_int_cpu)

    MV = p

    toc = time.time()
    time_eval += toc - tic
    
    return MV, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack, AI_int
