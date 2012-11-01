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

def gmres_dot (Precond, E_hat, X, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, 
                normal_xDev, normal_yDev, normal_zDev, xj, yj, zj, xi, yi, zi,
                xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost,
                xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev,
                sizeTarDev, offsetMltDev, offsetIntDev, intPtrDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost, 
                offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, Cells, theta, Nm, II, JJ, KK, 
                index, index_large, IndexDev, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, index_ptr, 
                P, kappa, NCRIT, K, threshold, BSZ, GSZ, BlocksPerTwig, twig, eps, time_eval, time_P2P, 
                time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans):

    N = len(triangle)
    MV = zeros(len(X))
    L = sqrt(2*Area) # Representative length
    AI_int = 0
    
    w    = getWeights(K)
    X_K = zeros(N*K, dtype=REAL)
    X_Kx = zeros(N*K, dtype=REAL)
    X_Ky = zeros(N*K, dtype=REAL)
    X_Kz = zeros(N*K, dtype=REAL)
    X_Kclean = zeros(N*K, dtype=REAL)
    
    for i in range(N*K):
        X_K[i]   =  X[i/K+N]*w[i%K]*Area[i/K]
        X_Kx[i]  = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,0]
        X_Ky[i]  = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,1]
        X_Kz[i]  = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,2]
        X_Kclean[i] = X[i/K]
    # The minus sign comes from the chain rule when deriving
    # with respect to r' in 1/|r-r'|

    tic = cuda.Event()
    toc = cuda.Event()
    tic.record()
    C = 0
    getMultipole(Cells, C, xj, yj, zj, X_K, X_Kx, X_Ky, X_Kz, II, JJ, KK, index, P, NCRIT)
#    getMultipole2(Cells, xj, yj, zj, X_K, X_Kx, X_Ky, X_Kz, II, JJ, KK, index, P, twig)
    toc.record()
    toc.synchronize()
    time_P2M += tic.time_till(toc)*1e-3

    tic = tic.record()
    for C in reversed(range(1,len(Cells))):
        PC = Cells[C].parent
        upwardSweep(Cells,C,PC,P, II, JJ, KK, index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, index_ptr)
    toc.record()
    toc.synchronize()
    time_M2M += tic.time_till(toc)*1e-3

    tic.record()
    mHost = X_K[srcPtr[0:offSrc]]
    mxHost = X_Kx[srcPtr[0:offSrc]]
    myHost = X_Ky[srcPtr[0:offSrc]]
    mzHost = X_Kz[srcPtr[0:offSrc]]
    mcleanHost = X_Kclean[srcPtr[0:offSrc]]

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
    toc.record()
    toc.synchronize()
    time_pack += tic.time_till(toc)*1e-3

    tec = cuda.Event()
    tac = cuda.Event()
    tic.record()
    

    if len(mpHost>0):
        mpDev = cuda.to_device(mpHost.astype(REAL)) 
        mpxDev = cuda.to_device(mpxHost.astype(REAL)) 
        mpyDev = cuda.to_device(mpyHost.astype(REAL)) 
        mpzDev = cuda.to_device(mpzHost.astype(REAL)) 

    mDev = cuda.to_device(mHost.astype(REAL)) 
    mxDev = cuda.to_device(mxHost.astype(REAL)) 
    myDev = cuda.to_device(myHost.astype(REAL)) 
    mzDev = cuda.to_device(mzHost.astype(REAL)) 
    mcleanDev = cuda.to_device(mcleanHost.astype(REAL)) 

#    (free,total) = cuda.mem_get_info()
#    print 'Global memory occupancy: %f%% free'%(free*100/total)

    toc.record()
    toc.synchronize()
    time_trans += tic.time_till(toc)*1e-3

    tic.record()

    p  = zeros(len(X))
#    p1Dev = gpuarray.zeros(len(xtHost), dtype=REAL)
#    p2Dev = gpuarray.zeros(len(xtHost), dtype=REAL)
    pHost = zeros(len(xtHost), dtype=REAL)
    p1Dev = cuda.to_device(pHost.astype(REAL))
    p2Dev = cuda.to_device(pHost.astype(REAL))


    xkSize = len(xk)

    tec.record()
    mod = kernels(BSZ,Nm,xkSize,P)
    M2P_gpu = mod.get_function("M2P")
    if len(mpHost>0):
        M2P_gpu(sizeTarDev, offsetMltDev, xtDev, ytDev, ztDev, xcDev, ycDev, zcDev, 
                mpDev, mpxDev, mpyDev, mpzDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, p1Dev, p2Dev, IndexDev,
                int32(N), REAL(kappa), REAL(E_hat), int32(BlocksPerTwig), int32(NCRIT), 
                block=(BSZ,1,1),grid=(GSZ,1))
    # M2P on CPU
#    p = M2P_pack(tarPtr, xtHost, ytHost, ztHost, offsetTarHost, sizeTarHost, 
#               xcHost, ycHost, zcHost, mpHost, mpxHost, mpyHost, mpzHost,
#               Pre0Host, Pre1Host, Pre2Host, Pre3Host,
#               offsetMltHost, p, P, kappa, Nm, N, Precond, E_hat)
    tac.record()
    tac.synchronize()
    time_M2P += tec.time_till(tac)*1e-3

    tec.record()
#    AI_int_gpu = gpuarray.zeros(len(xtHost), dtype=int32)
    AI_int_gpu = cuda.to_device(zeros(len(xtHost), dtype=int32))
    P2P_gpu = mod.get_function("P2P")
    P2P_gpu(offsetSrcDev, offsetIntDev, intPtrDev, sizeTarDev, kDev, 
           xsDev, ysDev, zsDev, mDev, mxDev, myDev, mzDev, mcleanDev, xtDev, ytDev, ztDev, 
            AreaDev, p1Dev, p2Dev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, REAL(E_hat), 
            int32(N), vertexDev, REAL(w[0]), xkDev, wkDev, 
            REAL(kappa), REAL(threshold), REAL(eps), int32(BlocksPerTwig), int32(NCRIT), 
            AI_int_gpu, block=(BSZ,1,1), grid=(GSZ,1))


    # P2P on CPU (not working)
#    p, AI_int, time_an = P2P_pack(xi, yi, zi, offsetSrcHost, offsetTarHost, sizeTarHost, 
#                                tarPtr, srcPtr, xsHost, ysHost, zsHost, mHost, mxHost, myHost, mzHost, 
#                                xtHost, ytHost, ztHost, E_hat, N, p, vertexHost, vertex, triangle, AreaHost, Area, normal[:,0], 
#                                triHost, kHost, K, w, xk, wk, Precond, kappa, threshold, eps, time_an, AI_int)
 
    tac.record()
    tac.synchronize()
    time_P2P += tec.time_till(tac)*1e-3

    toc.record()
    toc.synchronize()
    time_eval += tic.time_till(toc)*1e-3

    tic.record()
    paux1 = zeros(len(xtHost), dtype=REAL)
    paux2 = zeros(len(xtHost), dtype=REAL)
#    p1Dev.get(paux1)
#    p2Dev.get(paux2)
    paux1 = cuda.from_device(p1Dev, len(xtHost), dtype=REAL)
    paux2 = cuda.from_device(p2Dev, len(xtHost), dtype=REAL)
    p_test = zeros(len(p))
    AI_aux = zeros(len(xtHost), dtype=int32)
#    AI_int_gpu.get(AI_aux)
    AI_aux = cuda.from_device(AI_int_gpu, len(xtHost), dtype=int32)
    AI_int_cpu = zeros(len(p))
    toc.record()
    toc.synchronize()
    time_trans += tic.time_till(toc)*1e-3

    tic.record()
    t = -1
    for CI in twig:
        t += 1
        CI_start = offsetTarHost[t]
        CI_end = offsetTarHost[t] + sizeTarHost[t]
        targets = tarPtr[CI_start:CI_end]
        AI_int_cpu[targets] += AI_aux[CI_start:CI_end]
        p[targets]   += paux1[CI_start:CI_end]
        p[targets+N] += paux2[CI_start:CI_end]
    toc.record()
    toc.synchronize()
    time_pack += tic.time_till(toc)*1e-3

  
    '''
    p_test2 = zeros(len(X))
    for CI in twig:
        p_test2,time_M2P = M2P_list(Cells, CI, Precond, xi, yi, zi, p_test2, Nm, P, N, E_hat, kappa, eps, time_M2P)        
    for CI in twig:
        p_test2, AI_int, time_P2P, time_an = P2P_wrap(Cells, CI, Precond, xj,yj, zj, X_K, X_Kx, X_Ky, X_Kz, xi, yi, zi, p_test2, theta, Area, normal[:,0], kappa, eps, L, K, vertex, triangle, w, xk, wk, E_hat, threshold, time_P2P, time_an, AI_int)
    '''
    '''
    errorTest = sqrt(sum((p_test-p)*(p_test-p))/(sum(p*p)+1e-16))
    if errorTest>1e-12:
        print 'wrong'
        print p
        print p_test
    else:
        print 'right'
        print p
        print p_test
    '''

    AI_int = sum(AI_int_cpu)
    
    return p, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans, AI_int

def get_phir (X, vertex, triangle, xj, yj, zj, xi, yi, zi, xq, yq, zq,
            Area, normal, xk, wk, Cells, theta, Nm, II, JJ, KK, 
            index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, 
            P, kappa, NCRIT, K, threshold, eps):

    N = len(triangle)
    MV = zeros(len(X))
    L = sqrt(2*Area) # Representative length
    AI_int = 0
    
    # Setup vector
    tic = time.time()
    w    = getWeights(K)
    X_K = zeros(N*K, dtype=REAL)
    X_Kx = zeros(N*K, dtype=REAL)
    X_Ky = zeros(N*K, dtype=REAL)
    X_Kz = zeros(N*K, dtype=REAL)
    X_Kclean = zeros(N*K, dtype=REAL)
    
    for i in range(N*K):
        X_K[i] = X[i/K+N]*w[i%K]*Area[i/K]
        X_Kx[i]  = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,0]
        X_Ky[i]  = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,1]
        X_Kz[i]  = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,2]
        X_Kclean[i] = X[i/K]
    # The minus sign comes from the chain rule when deriving
    # with respect to r' in 1/|r-r'|
    toc = time.time()
    time_set = toc - tic

    # P2M
    tic = time.time()
    C = 0
    getMultipole(Cells, C, xj, yj, zj, X_K, X_Kx, X_Ky, X_Kz, II, JJ, KK, index, P, NCRIT)
    toc = time.time()
    time_P2M = toc - tic

    # M2M
    tic = time.time()
    for C in reversed(range(1,len(Cells))):
        PC = Cells[C].parent
        upwardSweep(Cells,C,PC,P, II, JJ, KK, index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small)
    toc = time.time()
    time_M2M = toc - tic

    # Evaluation
    AI_int = 0
    phi_reac = zeros(len(xq))
    time_P2P = 0.
    time_M2P = 0.
    for i in range(len(xq)):
        CJ = 0
        p = 0.
        source = []
        p, source, time_M2P = M2P_nonvec(Cells, CJ, xq[i], yq[i], zq[i],
                                        p, theta, Nm, P, kappa, NCRIT, source, time_M2P)
        p, AI_int, time_P2P = P2P_nonvec(xj, yj, zj, X_K, X_Kx, X_Ky, X_Kz, X_Kclean, 
                                        xi, yi, zi, xq[i], yq[i], zq[i], p, vertex, triangle, 
                                        Area, kappa, K, w, xk, wk, threshold, source, eps, AI_int, time_P2P)
        phi_reac[i] = p
    print '\tTime set: %f'%time_P2M
    print '\tTime P2M: %f'%time_P2M
    print '\tTime M2M: %f'%time_M2M
    print '\tTime M2P: %f'%time_M2P
    print '\tTime P2P: %f'%time_P2P

    print '%i of %i analytical integrals for phi_reac calculation'%(AI_int/len(xq),len(xi))

    return phi_reac


def get_phir_gpu (X, vertex, triangle, xj, yj, zj, xq, yq, zq,
                Area, normal, xk, wk, K, threshold, BSZ, mod):

    N  = len(triangle)
    Nq = len(xq)
    Nj = N*K
    MV = zeros(len(X))

    # Setup vector
    tic = time.time()
    w        = getWeights(K)
    X_K      = zeros(Nj, dtype=REAL)
    X_Kx     = zeros(Nj, dtype=REAL)
    X_Ky     = zeros(Nj, dtype=REAL)
    X_Kz     = zeros(Nj, dtype=REAL)
    X_Kclean = zeros(Nj, dtype=REAL)
    vertex_pack = zeros((Nj,3,3), dtype=REAL)
    
    for i in range(Nj):
        X_K[i]      = X[i/K+N]*w[i%K]*Area[i/K]
        X_Kx[i]     = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,0]
        X_Ky[i]     = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,1]
        X_Kz[i]     = -X[i/K]*w[i%K]*Area[i/K]*normal[i/K,2]
        X_Kclean[i] = X[i/K]
        vertex_pack[i] = vertex[triangle[i/K]]
    # The minus sign comes from the chain rule when deriving
    # with respect to r' in 1/|r-r'|

    k = arange(Nj, dtype=int32)%K

#    vertex_pack = ravel(vertex_pack)
    vertex_pack = ravel(vertex[triangle[:]])

    phir = gpuarray.zeros(Nq, dtype=REAL)
    m_gpu  = cuda.to_device(X_K.astype(REAL))
    mx_gpu = cuda.to_device(X_Kx.astype(REAL))
    my_gpu = cuda.to_device(X_Ky.astype(REAL))
    mz_gpu = cuda.to_device(X_Kz.astype(REAL))
    mc_gpu = cuda.to_device(X_Kclean.astype(REAL))
    xj_gpu = cuda.to_device(xj.astype(REAL))
    yj_gpu = cuda.to_device(yj.astype(REAL))
    zj_gpu = cuda.to_device(zj.astype(REAL))
    xq_gpu = cuda.to_device(xq.astype(REAL))
    yq_gpu = cuda.to_device(yq.astype(REAL))
    zq_gpu = cuda.to_device(zq.astype(REAL))
    A_gpu  = cuda.to_device(Area.astype(REAL))
    k_gpu  = cuda.to_device(k.astype(int32))
    xk_gpu = cuda.to_device(xk.astype(REAL))
    wk_gpu = cuda.to_device(wk.astype(REAL))
    vertex_gpu = cuda.to_device(vertex_pack.astype(REAL))
    AI_int_gpu = cuda.to_device(zeros(Nq, dtype=int32))

    GSZ = int(ceil(float(Nq)/BSZ))

#    mod = kernels(BSZ,1,len(xk))
    get_phir = mod.get_function("get_phir")
    get_phir(phir, xj_gpu, yj_gpu, zj_gpu, 
             m_gpu, mx_gpu, my_gpu, mz_gpu, mc_gpu,
             xq_gpu, yq_gpu, zq_gpu, A_gpu, k_gpu,
             vertex_gpu, int32(N), int32(Nj), int32(Nq),
             int32(K), REAL(w[0]), xk_gpu, wk_gpu, REAL(threshold), 
             AI_int_gpu, block=(BSZ,1,1), grid=(GSZ,1))
    
    AI_aux = zeros(Nq, dtype=int32)
    AI_aux = cuda.from_device(AI_int_gpu, Nq, dtype=int32)
    AI_int = sum(AI_aux)
    print '%i of %i analytical integrals for phi_reac calculation'%(AI_int/Nq, N)

    phir_cpu = zeros(Nq, dtype=REAL)
    phir.get(phir_cpu)

    return phir_cpu
    

