from numpy              import *
from numpy              import sum as npsum
from scipy.misc         import factorial
from scipy.misc.common  import comb

# Wrapped code
from direct_c           import direct_c
from getCoeffwrap       import multipole, getCoeff, getIndex, getIndex_arr
from calculateMultipoles import P2M, M2M

import sys
sys.path.append('../util')
from semi_analytical     import SA_arr
from semi_analyticalwrap import SA_wrap_arr, P2P_c
from integral_matfree    import AI_arr

import time

class Cell():
    def __init__ (self, NCRIT, Nm):
        self.nsource   = 0       # Number of source particles
        self.ntarget = 0       # Number of target particles
        self.nchild  = 0      # Number of child boxes in binary
                              # This will be a 8bit value and if certain 
                              # child exists, that bit will be 1.

        self.source   = array([], dtype=int32)        # Pointer to source particles
        self.target = array([], dtype=int32)        # Pointer to target particles
        self.xc = 0.                                # x position of cell
        self.yc = 0.                                # y position of cell
        self.zc = 0.                                # z position of cell
        self.r  = 0.                                # cell radius

        self.parent = 0                             # Pointer to parent cell
        self.child  = zeros(8, dtype=int32)         # Pointer to child cell

        self.M = zeros(Nm)                          # Array with multipoles
        self.Mx = zeros(Nm)                          # Array with multipoles for d/dx.n
        self.My = zeros(Nm)                          # Array with multipoles for d/dy.n
        self.Mz = zeros(Nm)                          # Array with multipoles for d/dz.n

        self.P2P_list = array([], dtype=int32)       # Pointer to cells that interact with P2P
        self.M2P_list = array([], dtype=int32)       # Pointer to cells that interact with M2P
        self.list_ready = 0                          # Flag to know if P2P list is already generated

def add_child(octant, Cells, i, NCRIT, Nm):
    # add_child adds child cell to Cells array
    # octant: octant of the child cell
    # Cells : arrays with cells
    # i     : index of parent cell in Cells array

    CN    = Cell(NCRIT, Nm) # CN: child cell
    CN.r  = Cells[i].r/2
    CN.xc = Cells[i].xc + CN.r*((octant&1)*2-1) # octant&X returns X if true
    CN.yc = Cells[i].yc + CN.r*((octant&2)-1)   # Want to make ((octant&X)*Y - Z)=1
    CN.zc = Cells[i].zc + CN.r*((octant&4)/2-1)
    CN.parent = i
    Cells[i].child[octant] = len(Cells)
    Cells[i].nchild|=(1<<octant)
    Cells.append(CN)

def split_cell(x, y, z, Cells, C, NCRIT, Nm):
    # split_cell splits cell with more than NCRIT particles
    # x,y,z: positions of particles
    # Cells: array of cells
    # C    : index of cell to be split in Cells array

    for l in Cells[C].target:
        octant = (x[l]>Cells[C].xc) + ((y[l]>Cells[C].yc) << 1) + ((z[l]>Cells[C].zc) << 2)
        if (not(Cells[C].nchild & (1<<octant))): # Ask if octant exists already
            add_child(octant, Cells, C, NCRIT, Nm)

        CC = Cells[C].child[octant] # Pointer to child cell
        Cells[CC].target = append(Cells[CC].target, l)
        Cells[CC].ntarget += 1

        if (Cells[CC].ntarget >= NCRIT):
            split_cell(x, y, z, Cells, CC, NCRIT, Nm)

def generateTree(xi, yi, zi, NCRIT, Nm, N, radius, x_center):

    C0 = Cell(NCRIT, Nm)
    C0.xc = x_center[0]
    C0.yc = x_center[1]
    C0.zc = x_center[2]
    C0.r  = radius + 0.1

    Cells = []
    Cells.append(C0)

    for i in range(N):

        C = 0 
        while (Cells[C].ntarget>=NCRIT):
            Cells[C].ntarget+=1

            octant = (xi[i]>Cells[C].xc) + ((yi[i]>Cells[C].yc) << 1) + ((zi[i]>Cells[C].zc) << 2)
            if (not(Cells[C].nchild & (1<<octant))):
                add_child(octant, Cells, C, NCRIT, Nm)
        
            C = Cells[C].child[octant]

        Cells[C].target = append(Cells[C].target, i) 
        Cells[C].ntarget += 1

        if (Cells[C].ntarget>=NCRIT):
            split_cell(xi,yi,zi,Cells,C, NCRIT, Nm)

    return Cells

def computeIndices(P):
    II = []
    JJ = []
    KK = []
    index = []
    index_large = zeros((P+1)*(P+1)*(P+1), dtype=int32)
    for ii in range(P+1):
        for jj in range(P+1-ii):
            for kk in range(P+1-ii-jj):
                index.append(getIndex(P,ii,jj,kk))
                II.append(ii)
                JJ.append(jj)
                KK.append(kk)
                index_large[(P+1)*(P+1)*ii+(P+1)*jj+kk] = index[-1]

    II = array(II,int32)
    JJ = array(JJ,int32)
    KK = array(KK,int32)
    index = array(index,int32)
#    index = getIndex_arr(P,II,JJ,KK)
    
    return II, JJ, KK, index, index_large

def findTwigs(Cells, C, twig, NCRIT):
    # Cells     : array of cells
    # C         : index of cell in Cells array 
    # twig      : array with indices of twigs in Cells array

    if (Cells[C].ntarget>=NCRIT):
        for c in range(8):
            if (Cells[C].nchild & (1<<c)):
                twig = findTwigs(Cells, Cells[C].child[c], twig, NCRIT)
    else:
        twig.append(C)

    return twig

def getMultipole(Cells, C, x, y, z, m, mx, my, mz, II, JJ, KK, index, P, NCRIT):
    # Cells     : array of cells
    # C         : index of cell in Cells array 
    # x,y,z     : position of particles
    # m         : weight of particles
    # P         : order of Taylor expansion
    # NCRIT     : max number of source particles per cell
    # II,JJ,KK  : x,y,z powers of multipole expansion
    # index     : 1D mapping of II,JJ,KK (index of multipoles)

    if (Cells[C].ntarget>=NCRIT):

        Cells[C].M[:] = 0.0 # Initialize multipoles
        Cells[C].Mx[:] = 0.0
        Cells[C].My[:] = 0.0
        Cells[C].Mz[:] = 0.0

        for c in range(8):
            if (Cells[C].nchild & (1<<c)):
                getMultipole(Cells, Cells[C].child[c], x, y, z, m, mx, my, mz, II, JJ, KK, index, P, NCRIT)
    else:

        Cells[C].M[:] = 0.0 # Initialize multipoles
        Cells[C].Mx[:] = 0.0
        Cells[C].My[:] = 0.0
        Cells[C].Mz[:] = 0.0

        l = Cells[C].source
        P2M(Cells[C].M, Cells[C].Mx, Cells[C].My, Cells[C].Mz, x[l], y[l], z[l], m[l], mx[l], my[l], mz[l], Cells[C].xc, Cells[C].yc, Cells[C].zc, II, JJ, KK)
    

def addSources(Cells, twig, K):
    # This version of addSources puts the sources in the same cell
    # as the collocation point of the same panel
    
    for C in twig:
        Cells[C].nsource = K*Cells[C].ntarget
        for j in range(K):
            Cells[C].source = append(Cells[C].source, K*Cells[C].target + j)
        
def precompute_terms(P, II, JJ, KK): 
    # Precompute terms for 
    combII = array([],dtype=int32)
    combJJ = array([],dtype=int32)
    combKK = array([],dtype=int32)
    IImii  = array([],dtype=int32)
    JJmjj  = array([],dtype=int32)
    KKmkk  = array([],dtype=int32)
    index  = array([], dtype=int32)
    index_ptr = zeros(len(II)+1, dtype=int32)
    for i in range(len(II)):
        ii,jj,kk = mgrid[0:II[i]+1:1,0:JJ[i]+1:1,0:KK[i]+1:1].astype(int32)
        ii,jj,kk = ii.ravel(), jj.ravel(), kk.ravel()
        index_aux = zeros(len(ii), int32)
        getIndex_arr(P, len(ii), index_aux, ii, jj, kk)  
        index = append(index, index_aux)
        index_ptr[i+1] = len(index_aux)+index_ptr[i]
        combII = append(combII, comb(II[i],ii))
        combJJ = append(combJJ, comb(JJ[i],jj))
        combKK = append(combKK, comb(KK[i],kk))
        IImii = append(IImii, II[i]-ii)
        JJmjj = append(JJmjj, JJ[i]-jj)
        KKmkk = append(KKmkk, KK[i]-kk)
    return combII, combJJ, combKK, IImii, JJmjj, KKmkk, index, index_ptr

   
def upwardSweep(Cells, CC, PC, P, II, JJ, KK, Index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index, index_ptr):
    # Cells     : array of cells
    # CC        : index of child cell in Cells array
    # PC        : index of parent cell in Cells array
    # P         : order of Taylor expansion
    # II,JJ,KK  : x,y,z powers of multipole expansion
    # index     : 1D mapping of II,JJ,KK (index of multipoles)

    dx = Cells[PC].xc - Cells[CC].xc
    dy = Cells[PC].yc - Cells[CC].yc
    dz = Cells[PC].zc - Cells[CC].zc

    M2M(Cells[PC].M, Cells[CC].M, dx, dy, dz, II, JJ, KK, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index, index_ptr)
    M2M(Cells[PC].Mx, Cells[CC].Mx, dx, dy, dz, II, JJ, KK, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index, index_ptr)
    M2M(Cells[PC].My, Cells[CC].My, dx, dy, dz, II, JJ, KK, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index, index_ptr)
    M2M(Cells[PC].Mz, Cells[CC].Mz, dx, dy, dz, II, JJ, KK, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index, index_ptr)



def packData(Cells, CJ, CI, intPtr, mltPtr, offInt, offMlt, theta, NCRIT):
    # Cells     : array of Cells
    # CJ        : index of source cell
    # CI        : index of target cell
    # srcPtr    : array with pointers to sources 
    # mltPtr    : array with pointers to cells for M2P 
    # offSrc    : array with offsets to sources
    # offMlt    : array with offsets to cells for M2P
    # theta     : MAC criteron 
    # NCRIT     : max number of particles per cell

    if (Cells[CJ].ntarget>=NCRIT):
        for c in range(8):
            if (Cells[CJ].nchild & (1<<c)):
                CC = Cells[CJ].child[c]  # Points at child cell
                dxi = Cells[CC].xc - Cells[CI].xc
                dyi = Cells[CC].yc - Cells[CI].yc
                dzi = Cells[CC].zc - Cells[CI].zc
                r   = sqrt(dxi*dxi+dyi*dyi+dzi*dzi)
                if Cells[CI].r+Cells[CC].r > theta*r: # Max distance between particles
                    intPtr, mltPtr, offInt, offMlt = packData(Cells, CC, CI, intPtr, mltPtr, offInt, offMlt, theta, NCRIT)
                else:
#                    mltPtr = append(mltPtr, CC)
                    mltPtr[offMlt] = CC
                    offMlt += 1

    else: # Else on a twig cell
#        intPtr[offInt:offInt+Cells[CJ].nsource] = Cells[CJ].source
        intPtr[offInt] = CJ
        offInt += 1 

    return intPtr, mltPtr, offInt, offMlt

def M2P_pack(tarPtr, xtHost, ytHost, ztHost, offsetTarHost, sizeTarHost, 
           xcHost, ycHost, zcHost, mpHost, mpxHost, mpyHost, mpzHost, 
           Pre0Host, Pre1Host, Pre2Host, Pre3Host,
           offsetMltHost, p, P, kappa, Nm, N, Precond, E_hat):

    for off in range(len(offsetTarHost)):

        CJ_start = offsetMltHost[off]
        CJ_end   = offsetMltHost[off+1]
        CI_start = offsetTarHost[off]
        CI_end   = offsetTarHost[off] + sizeTarHost[off]
        xi = xtHost[CI_start:CI_end]
        yi = ytHost[CI_start:CI_end]
        zi = ztHost[CI_start:CI_end]
        P0 = Pre0Host[CI_start:CI_end]
        P1 = Pre1Host[CI_start:CI_end]
        P2 = Pre2Host[CI_start:CI_end]
        P3 = Pre3Host[CI_start:CI_end]
        
        M  = zeros(Nm)
        Mx = zeros(Nm)
        My = zeros(Nm)
        Mz = zeros(Nm)

        targets  = tarPtr[CI_start:CI_end]
        ntargets = CI_end - CI_start
        for CJ in range(CJ_start,CJ_end):
            dxi = xi - xcHost[CJ]
            dyi = yi - ycHost[CJ]
            dzi = zi - zcHost[CJ]

            L  = zeros(ntargets)
            dL = zeros(ntargets)
            Y  = zeros(ntargets)
            dY = zeros(ntargets)

            M  = mpHost [CJ*Nm:CJ*Nm+Nm]
            Mx = mpxHost[CJ*Nm:CJ*Nm+Nm]
            My = mpyHost[CJ*Nm:CJ*Nm+Nm]
            Mz = mpzHost[CJ*Nm:CJ*Nm+Nm]

            multipole(L, dL, Y, dY, M, Mx, My, Mz, dxi, dyi, dzi, P, kappa, int(Nm), E_hat)

            p[targets] += P0*(-dL+L) + P1*(dY-Y)
            p[targets+N] += P2*(-dL+L) + P3*(dY-Y)

    return p

def P2P_pack(xi, yi, zi, offsetSrcHost, offsetTarHost, sizeTarHost, tarPtr, srcPtr, 
            xsHost, ysHost, zsHost, mHost, mxHost, myHost, mzHost, xtHost, ytHost, ztHost, 
            E_hat, N, p, vertexHost, vertex, triangle, AreaHost, Area, normal_x, 
            triHost, kHost, K, w, xk, wk, Precond, kappa, threshold, eps, time_an, AI_int):

    for off in range(len(offsetTarHost)):

        CI_start = offsetTarHost[off]
        CI_end   = offsetTarHost[off] + sizeTarHost[off]
        xt = xtHost[CI_start:CI_end]
        yt = ytHost[CI_start:CI_end]
        zt = ztHost[CI_start:CI_end]

        CJ_start = offsetSrcHost[off]
        CJ_end   = offsetSrcHost[off+1]
        s_xj = xsHost[CJ_start:CJ_end]
        s_yj = ysHost[CJ_start:CJ_end]
        s_zj = zsHost[CJ_start:CJ_end]
        s_m = mHost[CJ_start:CJ_end]
        s_mx = mxHost[CJ_start:CJ_end]
        s_my = myHost[CJ_start:CJ_end]
        s_mz = mzHost[CJ_start:CJ_end]

        s_vertex = vertexHost[CJ_start*9:CJ_end*9]
        s_A = AreaHost[CJ_start:CJ_end]
        s_tri = triHost[CJ_start:CJ_end]
        s_k = kHost[CJ_start:CJ_end]

        tri = srcPtr[CJ_start:CJ_end]/K # Triangle
        k = srcPtr[CJ_start:CJ_end]%K   # Gauss point

        # Wrapped P2P
        aux = zeros(2)
        ntargets = CI_end-CI_start
        targets  = tarPtr[CI_start:CI_end] 
        ML  = zeros(ntargets)
        dML = zeros(ntargets)
        MY  = zeros(ntargets)
        dMY = zeros(ntargets)
        P2P_c(MY, dMY, ML, dML, ravel(vertex[triangle[:]]), int32(s_tri), int32(s_k), 
            xi, yi, zi, s_xj, s_yj, s_zj, xt, yt, zt, s_m, s_mx, s_my, s_mz, targets,
            Area, normal_x, xk, wk, kappa, threshold, eps, w[0], aux)
        AI_int += int(aux[0])
        time_an += aux[1]

        # With preconditioner
        p[tarPtr[CI_start:CI_end]] += Precond[0,targets]*(dML+ML) - Precond[1,targets]*(dMY+E_hat*MY)
        p[tarPtr[CI_start:CI_end]+N] += Precond[2,targets]*(dML+ML) - Precond[3,targets]*(dMY+E_hat*MY)
        # No preconditioner
#        p[tarPtr[CI_start:CI_end]] += -dML+ML
#        p[tarPtr[CI_start:CI_end]+N] += dMY-E_hat*MY

    return p, AI_int, time_an

def M2P_nonvec(Cells, CJ, xq, yq, zq, p, theta, Nm, P, 
                      kappa, NCRIT,source, time_M2P):
    # Cells     : array of Cells
    # CJ        : index of source cell
    # p         : accumulator 
    # theta     : MAC criteron 
    # Nm        : Number of multipole coefficients
    # P         : order of Taylor expansion
    # NCRIT     : max number of particles per cell

    if (Cells[CJ].ntarget>=NCRIT): # if not a twig
        for c in range(8):
            if (Cells[CJ].nchild & (1<<c)):
                CC = Cells[CJ].child[c]  # Points at child cell
                dxi = Cells[CC].xc - xq
                dyi = Cells[CC].yc - yq 
                dzi = Cells[CC].zc - zq
                r   = sqrt(dxi*dxi+dyi*dyi+dzi*dzi)
                if Cells[CC].r > theta*r: # Max distance between particles
                    p, source,  time_M2P = M2P_nonvec(Cells, CC, xq, yq, zq, p, theta, 
                                                    Nm, P, kappa, NCRIT, source, time_M2P)
                else:
                    tic = time.time()
                    dxi = xq - Cells[CC].xc
                    dyi = yq - Cells[CC].yc
                    dzi = zq - Cells[CC].zc

                    L = zeros(1)
                    dL = zeros(1)
                    Y = zeros(1)
                    dY = zeros(1)
                    dxi = array([dxi])
                    dyi = array([dyi])
                    dzi = array([dzi])
                    multipole(L, dL, Y, dY, Cells[CC].M, Cells[CC].Mx, Cells[CC].My, 
                                Cells[CC].Mz, dxi, dyi, dzi, P, kappa, int(Nm), 1)

                    p += -dL+L
                    toc = time.time()
                    time_M2P += toc - tic

    else: # Else on a twig cell
        source.extend(Cells[CJ].source)

    return p, source, time_M2P

def P2P_nonvec(xj, yj, zj, m, mx, my, mz, mc, xi, yi, zi, 
                xq, yq, zq, p, vertex, triangle, Area, kappa, 
                K, w, xk, wk, threshold, source, eps, AI_int, time_P2P):

    tic = time.time()
    source = int32(array(source))
    s_xj = xj[source]
    s_yj = yj[source]
    s_zj = zj[source]
    s_m  = m[source]
    s_mx = mx[source]
    s_my = my[source]
    s_mz = mz[source]
    s_mc = mc[source]

    tri  = source/K # Triangle
    k    = source%K # Gauss point

    L  = zeros(1)
    dL = zeros(1)
    Y  = zeros(1)
    dY = zeros(1)

    xq_arr = array([xq])
    yq_arr = array([yq])
    zq_arr = array([zq])
    target = array([-1], dtype=int32)

    aux = zeros(2)
    P2P_c(Y, dY, L, dL, ravel(vertex[triangle[:]]), int32(tri), int32(k), 
        xi, yi, zi, s_xj, s_yj, s_zj, xq_arr, yq_arr, zq_arr, 
        s_m, s_mx, s_my, s_mz, s_mc, target, Area, xk, wk, kappa, threshold, eps, w[0], aux)
    AI_int += int(aux[0])


    p += -dL+L
    toc = time.time()
    time_P2P += toc - tic

    return p, AI_int, time_P2P
