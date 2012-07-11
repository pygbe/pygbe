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

from numpy              import *
from numpy              import sum as npsum
from scipy.misc         import factorial
from scipy.misc.common  import comb

# Wrapped code
#from direct_c           import direct_c
from getCoeffwrap       import multipole, getCoeff, getIndex, getIndex_arr

import sys
sys.path.append('../util')
#from semi_analytical     import SA_arr
#from semi_analyticalwrap import SA_wrap_arr, P2P_c

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

def generateTree(xi, yi, zi, NCRIT, Nm, N, radius):
    # xi, yi, zi: position of target 
    # NCRIT     : max number of particles per twig cell
    # Nm        : number of terms in Taylor expansion
    # N         : number of points
    # radius    : radius of sphere


    C0 = Cell(NCRIT, Nm)
    C0.xc = 1e-5
    C0.yc = 1e-5
    C0.zc = 1e-5
    C0.r  = radius + 0.1

    Cells = []
    Cells.append(C0)

	# Loop over targets
    for i in range(N):

        C = 0 
		# Traverse tree
        while (Cells[C].ntarget>=NCRIT):
            Cells[C].ntarget+=1

            octant = (xi[i]>Cells[C].xc) + ((yi[i]>Cells[C].yc) << 1) + ((zi[i]>Cells[C].zc) << 2)
            if (not(Cells[C].nchild & (1<<octant))):
                add_child(octant, Cells, C, NCRIT, Nm)
        
            C = Cells[C].child[octant]

		# Add target
        Cells[C].target = append(Cells[C].target, i) 
        Cells[C].ntarget += 1

		# If too many targets, split cell
        if (Cells[C].ntarget>=NCRIT):
            split_cell(xi,yi,zi,Cells,C, NCRIT, Nm)

    return Cells

def computeIndices(P):
    II = []
    JJ = []
    KK = []
    index = []
    for ii in range(P+1):
        for jj in range(P+1-ii):
            for kk in range(P+1-ii-jj):
                index.append(getIndex(P,ii,jj,kk))
                II.append(ii)
                JJ.append(jj)
                KK.append(kk)
    II = array(II,int32)
    JJ = array(JJ,int32)
    KK = array(KK,int32)
    index = array(index,int32)
#    index = getIndex_arr(P,II,JJ,KK)
    
    return II, JJ, KK, index

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

	# Traverse tree
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

        for l in Cells[C].source:
            dx = Cells[C].xc - x[l]
            dy = Cells[C].yc - y[l]
            dz = Cells[C].zc - z[l]

            Cells[C].M[index]  += m[l] *(dx**II)*(dy**JJ)*(dz**KK)
            Cells[C].Mx[index] += mx[l]*(dx**II)*(dy**JJ)*(dz**KK)
            Cells[C].My[index] += my[l]*(dx**II)*(dy**JJ)*(dz**KK)
            Cells[C].Mz[index] += mz[l]*(dx**II)*(dy**JJ)*(dz**KK)


def addSources2(x,y,z,j,Cells,C,NCRIT):
    # x,y,z: location of sources
    # j    : index of source
    # Cells: array with cells
    # C    : index of cell in Cells array
    # NCRIT: max number of source particles per cell

    if (Cells[C].ntarget>=NCRIT):
        octant = (x[i]>Cells[C].xc) + ((y[i]>Cells[C].yc) << 1) + ((z[i]>Cells[C].zc) << 2)
        if (Cells[C].nchild & (1<<octant)): # If child cell exists, use
            O = octant
        else:                               # If child cell doesn't exist add to closest existing child
            r = []
            child = []
            for c in range(8):
                if (Cells[C].nchild & (1<<c)):
                    dx = x[j]-Cells[Cells[C].child[c]].xc
                    dy = y[j]-Cells[Cells[C].child[c]].yc
                    dz = z[j]-Cells[Cells[C].child[c]].zc
                    r.append(sqrt(dx*dx+dy*dy+dz*dz))
                    child.append(c)
            close_child = r.index(min(r))   # Find index of closest child
            O = child[close_child]              
        addSources2(x,y,z,j,Cells, Cells[C].child[O],NCRIT)
        
    else:
        Cells[C].nsource += 1
        Cells[C].source = append(Cells[C].source, j)

def addSources(x,y,z,Cells,twig):
    # x,y,z: location of sources
    # Cells: array with cells
    # twig : array with pointers to twigs of cells array

    dx = zeros((len(twig),len(x)))
    dy = zeros((len(twig),len(x)))
    dz = zeros((len(twig),len(x)))
    j = 0
    for t in twig:
        dx[j] = x - Cells[t].xc
        dy[j] = y - Cells[t].yc
        dz[j] = z - Cells[t].zc
        j+=1
    r = sqrt(dx*dx+dy*dy+dz*dz)

    close_twig = argmin(r,axis=0)

    for j in range(len(close_twig)):
        Cells[twig[close_twig[j]]].nsource += 1
        Cells[twig[close_twig[j]]].source = append(Cells[twig[close_twig[j]]].source, j)


def precompute_terms(P, II, JJ, KK):
    # Precompute terms for upwardSweep
    combII = []
    combJJ = []
    combKK = []
    IImii  = []
    JJmjj  = []
    KKmkk  = []
    index  = [] 
    for i in range(len(II)):
        ii,jj,kk = mgrid[0:II[i]+1:1,0:JJ[i]+1:1,0:KK[i]+1:1].astype(int32)
        ii,jj,kk = ii.ravel(), jj.ravel(), kk.ravel()
        index_aux = zeros(len(ii), int32)
        getIndex_arr(P, len(ii), index_aux, ii, jj, kk)
        index.append(index_aux)
        combII.append(comb(II[i],ii))
        combJJ.append(comb(JJ[i],jj))
        combKK.append(comb(KK[i],kk))
        IImii.append(II[i]-ii)
        JJmjj.append(JJ[i]-jj)
        KKmkk.append(KK[i]-kk)
    return combII, combJJ, combKK, IImii, JJmjj, KKmkk, index
        

   
def upwardSweep(Cells, CC, PC, P, II, JJ, KK, Index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index):
    # Cells     : array of cells
    # CC        : index of child cell in Cells array
    # PC        : index of parent cell in Cells array
    # P         : order of Taylor expansion
    # II,JJ,KK  : x,y,z powers of multipole expansion
    # index     : 1D mapping of II,JJ,KK (index of multipoles)

    dx = Cells[PC].xc - Cells[CC].xc
    dy = Cells[PC].yc - Cells[CC].yc
    dz = Cells[PC].zc - Cells[CC].zc

    for ij in range(len(Index)):
        Cells[PC].M[Index[ij]] += npsum(Cells[CC].M[index[ij]]*combII[ij]*combJJ[ij]*combKK[ij]*dx**IImii[ij]*dy**JJmjj[ij]*dz**KKmkk[ij])
        Cells[PC].Mx[Index[ij]] += npsum(Cells[CC].Mx[index[ij]]*combII[ij]*combJJ[ij]*combKK[ij]*dx**IImii[ij]*dy**JJmjj[ij]*dz**KKmkk[ij])
        Cells[PC].My[Index[ij]] += npsum(Cells[CC].My[index[ij]]*combII[ij]*combJJ[ij]*combKK[ij]*dx**IImii[ij]*dy**JJmjj[ij]*dz**KKmkk[ij])
        Cells[PC].Mz[Index[ij]] += npsum(Cells[CC].Mz[index[ij]]*combII[ij]*combJJ[ij]*combKK[ij]*dx**IImii[ij]*dy**JJmjj[ij]*dz**KKmkk[ij])



def interaction_list(Cells,CJ,CI,theta,NCRIT):
    # Cells     : array of Cells
    # CJ        : index of source cell
    # CI        : index of target cell
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
                    interaction_list(Cells,CC,CI,theta,NCRIT)
                else:
                    Cells[CI].M2P_list = append(Cells[CI].M2P_list,CC)
    elif Cells[CI].list_ready == 0: # Else on a twig cell
    # Generate P2P interaction list only the first time we traverse the tree
        Cells[CI].P2P_list = append(Cells[CI].P2P_list,CJ)



def packData(Cells, CJ, CI, srcPtr, mltPtr, offSrc, offMlt, theta, NCRIT):
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
                    srcPtr, mltPtr, offSrc, offMlt = packData(Cells, CC, CI, srcPtr, mltPtr, offSrc, offMlt, theta, NCRIT)
                else:
#                    mltPtr = append(mltPtr, CC)
                    mltPtr[offMlt] = CC
                    offMlt += 1

    else: # Else on a twig cell
#        srcPtr = append(srcPtr, Cells[CJ].source)
        srcPtr[offSrc:offSrc+Cells[CJ].nsource] = Cells[CJ].source
        offSrc += Cells[CJ].nsource

    return srcPtr, mltPtr, offSrc, offMlt

def M2P_pack(tarPtr, xtHost, ytHost, ztHost, offsetTarHost, sizeTarHost, 
           xcHost, ycHost, zcHost, mpHost, mpxHost, mpyHost, mpzHost, 
           Pre0Host, Pre1Host, Pre2Host, Pre3Host,
           offsetMltHost, p, P, kappa, Nm, N, Precond, E_hat):
	# tarPtr       	: packed array with pointers to targets in xi, yi, zi array
	# xt,yt,ztHost 	: packed array with position of targets (collocation points) on host memory
	# offsetTarHost : array with pointers to first target of each twig in tarPtr
	# sizeTarHost   : array with number of targets per twig cell on host memory
	# xc,yc,zcHost	: packed array with position of box centers on host memory
	# mp,x,y,zHost	: packed array with values of multipoles relevant for 0,x,y,z derivatives
	# Pre0,1,2,3Host: packed array with diagonal values of preconditioner for blocks 0,1,2,3 on host
	# offsetMltHost : array with pointers to first element of each twig in xcHost/Dev array on host memory
	# p				: accumulator
	# P				: order of expansion
	# kappa			: inverse of Debye length
	# Nm			: number of terms of Taylor expansion
	# N				: number of elements
	# Precond		: preconditioner
	# E_hat			: coefficient of bottom right block



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

            p[targets] += P0*(dL+L) - P1*(dY+Y)
            p[targets+N] += P2*(dL+L) - P3*(dY+Y)

    return p

def P2P_pack(xi, yi, zi, offsetSrcHost, offsetTarHost, sizeTarHost, tarPtr, srcPtr, 
            xsHost, ysHost, zsHost, mHost, mxHost, myHost, mzHost, xtHost, ytHost, ztHost, 
            E_hat, N, p, vertexHost, vertex, triangle, AreaHost, Area, normal_x, 
            triHost, kHost, K, w, xk, wk, Precond, kappa, threshold, eps, time_an, AI_int):
	# tarPtr       	: packed array with pointers to targets in xi, yi, zi array
	# xt,yt,ztHost 	: packed array with position of targets (collocation points) on host memory
	# offsetTarHost : array with pointers to first target of each twig in tarPtr
	# sizeTarHost   : array with number of targets per twig cell on host memory
	# xc,yc,zcHost	: packed array with position of box centers on host memory
	# mp,x,y,zHost	: packed array with values of multipoles relevant for 0,x,y,z derivatives
	# Pre0,1,2,3Host: packed array with diagonal values of preconditioner for blocks 0,1,2,3 on host
	# offsetMltHost : array with pointers to first element of each twig in xcHost/Dev array on host memory
	# p				: accumulator
	# P				: order of expansion
	# kappa			: inverse of Debye length
	# Nm			: number of terms of Taylor expansion
	# N				: number of elements
	# Precond		: preconditioner
	# E_hat			: coefficient of bottom right block

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

        p[tarPtr[CI_start:CI_end]] += Precond[0,targets]*(dML+ML) - Precond[1,targets]*(dMY+E_hat*MY)
        p[tarPtr[CI_start:CI_end]+N] += Precond[2,targets]*(dML+ML) - Precond[3,targets]*(dMY+E_hat*MY)

    return p, AI_int, time_an
