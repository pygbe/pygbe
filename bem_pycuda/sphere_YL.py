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

"""
Spherical molecule with random charge distribution
Yoon and Lenhoff 1990 formulation

Analytical solution on Kirkwood 1934 
"""

from pylab 			import *
from math  			import pi
from scipy.misc     import factorial
from numpy          import float64 as REAL
import time

# Import self made modules
import sys 
from gmres			    import gmres_solver    
from matrixfree         import getGaussPoints

sys.path.append('../util')
from triangulation 	 import *
from an_solution     import an_P
from semi_analytical import *
#from semi_analyticalwrap import SA_wrap_arr

sys.path.append('tree')
from FMMutils import *

# pyCUDA libraries
import pycuda.autoinit
import pycuda.gpuarray  as gpuarray
import pycuda.driver    as cuda
#from cuda_kernels import kernels


Rec = 6		# Number of recursions to generate sphere
R   = 4. 	# Radius of sphere
K   = 1  	# Number of gauss points

threshold = 0.5 	# L/d threshold to change to analytical
                    # Over: analytical, under: quadrature

### Physical variables
qe = 1.60217646e-19             # Electron charge 
Nq = 1                          # Number of charges
q  = array([1.])                # Charge in electron charge units
xq = array([[1e-10,-1e-10,0.]])  # Locations of charges
Na = 6.0221415e23               # Avogadro's number
E_0 = 8.854187818e-12           # Vacuum dielectric constant
E_1 = 4.                        # Molecule dielectric constant
E_2 = 80.                       # Water dielectric constant
E_hat = E_1/E_2
kappa = 0.125                   # Screening parameter


### GMRES related variables
restart  = 100 	# Iterations before restart for GMRES
tol		 = 1e-5 # tolerance of GMRES
max_iter = 1000 # maximum number of iterations for GMRES

### Tree code related variables
P     = 6                                         # Order of Taylor expansion
eps   = 1e-10                                     # Epsilon machine
Nm    = int32(factorial(P+3)/(6*factorial(P)))    # Number of terms in Taylor expansion
NCRIT = 1000                                      # Max number of particles per twig cell
theta = 0.5                                       # MAC criterion
BSZ   = 256                                       # CUDA block size


### Setup elements
vertex, triangle, center = create_unit_sphere(Rec)
vertex *= R
center *= R
normal, Area = surfaceVariables(vertex,triangle)
# Targets
xi = center[:,0]
yi = center[:,1]
zi = center[:,2]

# Set Gauss points (sources)
xj,yj,zj = getGaussPoints(vertex,triangle,K)

N 		= len(triangle) # Number of elements (and targets)
Nj      = N*K           # Number of gauss points (sources)
bc     	= zeros(N)
print("**** "+str(N)+" elements ****")

Nk = 3          		# Number of Gauss points per side for semi-analytical integrals
xk,wk = GQ_1D(Nk) 		# 1D Gauss points position and weight

### Generate tree, compute indices and precompute terms for M2M
Cells = generateTree(xj,yj,zj,NCRIT,Nm,Nj,R)
twig = []
C = 0
twig = findTwigs(Cells, C, twig, NCRIT)

II,JJ,KK,index = computeIndices(P)

addSources(xi,yi,zi,Cells,twig)

combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small = precompute_terms(P, II, JJ, KK)

### Generate preconditioner
# Will use block-diagonal preconditioner (AltmanBardhanWhiteTidor2008)
dX11 = zeros(N) # Top left block
dX12 = zeros(N) # Top right block
dX21 = zeros(N) # Bottom left block
dX22 = zeros(N) # Bottom right block
Precond = zeros((4,N))  # Stores the inverse of the block diagonal (also a tridiag matrix)
                        # Order: Top left, top right, bott left, bott right    

'''
for i in range(N):
    panel = vertex[triangle[i]]
    center = array([xi[i],yi[i],zi[i]])
    same = array([1], dtype=int32)
    Aaux = zeros(1) # Top left
    Baux = zeros(1) # Top right
    Caux = zeros(1) # Bottom left
    Daux = zeros(1) # Bottom right

    SA_wrap_arr(ravel(panel), center, Daux, Caux, Baux, Aaux, kappa, same, xk, wk)
    dX11[i] = Aaux
    dX12[i] = Baux
    dX21[i] = -Caux
    dX22[i] = -E_hat*Daux

d_aux = 1/(dX22-dX21*dX12/dX11)
Precond[0,:] = 1/dX11 + 1/dX11*dX12*d_aux*dX21/dX11
Precond[1,:] = -1/dX11*dX12*d_aux
Precond[2,:] = -d_aux*dX21/dX11
Precond[3,:] = d_aux
'''

tic = time.time()

### Output parameters
rr = zeros(len(Cells))
for i in range(len(Cells)):
    rr[i] = Cells[i].r
Levels = log(Cells[0].r/min(rr))/log(2) + 1

print 'Cells : %i'%len(Cells)
print 'Twigs : %i'%len(twig)
print 'Levels: %i'%Levels
print 'Twig cell size   : %f'%(min(rr))
print 'Rbox/theta       : %f'%(min(rr)/theta)
print 'Analytic distance: %f'%(average(sqrt(2*Area))/threshold)

### Generate RHS
tic = time.time()
dx_pq = zeros((Nq,N))
dy_pq = zeros((Nq,N))
dz_pq = zeros((Nq,N))
for i in range(Nq):
        dx_pq[i,:] = xi - xq[i,0] 
        dy_pq[i,:] = yi - xq[i,1] 
        dz_pq[i,:] = zi - xq[i,2] 

R_pq  = sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)

F = zeros(2*N)
# With preconditioner
#F[0:N] 		= sum(-transpose(q*ones((N,Nq)))/(E_1*R_pq),axis=0) * Precond[0,:]
#F[N:2*N] 	= sum(-transpose(q*ones((N,Nq)))/(E_1*R_pq),axis=0) * Precond[2,:]
# No preconditioner
F[0:N] 		= sum(-transpose(q*ones((N,Nq)))/(E_1*R_pq),axis=0)
toc = time.time()
rhs_time = toc-tic
print 'RHS generation time: %fs\n'%rhs_time

### Data packing
tic = time.time()
Nround = len(twig)*NCRIT
Nlist  = Nround*len(twig)

# Initialize packed arrays
offsetTarHost = zeros(len(twig)  , dtype=int32)
sizeTarHost   = zeros(len(twig)  , dtype=int32)
offsetSrcHost = zeros(len(twig)+1, dtype=int32)
offsetMltHost = zeros(len(twig)+1, dtype=int32)

tarPtr = zeros(Nround, dtype=int32)
srcPtr = zeros(Nlist , dtype=int32)
mltPtr = zeros(len(twig)*len(twig), dtype=int32)

xtHost = zeros(Nround)
ytHost = zeros(Nround)
ztHost = zeros(Nround)

Pre0Host = zeros(Nround)
Pre1Host = zeros(Nround)
Pre2Host = zeros(Nround)
Pre3Host = zeros(Nround)

# Packi pointers of sources and multipoles
t = -1
offSrc = 0 
offMlt = 0 
for CI in twig:
    t += 1
    offsetSrcHost[t] = offSrc
    offsetMltHost[t] = offMlt
    CJ = 0 
    srcPtr, mltPtr, offSrc, offMlt = packData(Cells, CJ, CI, srcPtr, mltPtr, offSrc, offMlt, theta, NCRIT)
offsetSrcHost[-1] = offSrc
offsetMltHost[-1] = offMlt

# Pack targets and preconditioner
twig_n = -1
for CI in twig:
    twig_n += 1
    targets = Cells[CI].target
    ntargets = Cells[CI].ntarget

    offsetTarHost[twig_n] = twig_n*NCRIT
    sizeTarHost[twig_n] = ntargets

    tarPtr[twig_n*NCRIT:twig_n*NCRIT+Cells[CI].ntarget] = targets
    xtHost[twig_n*NCRIT:twig_n*NCRIT+ntargets] = xi[targets]
    ytHost[twig_n*NCRIT:twig_n*NCRIT+ntargets] = yi[targets]
    ztHost[twig_n*NCRIT:twig_n*NCRIT+ntargets] = zi[targets]

    Pre0Host[twig_n*NCRIT:twig_n*NCRIT+ntargets] = Precond[0,targets]
    Pre1Host[twig_n*NCRIT:twig_n*NCRIT+ntargets] = Precond[1,targets]
    Pre2Host[twig_n*NCRIT:twig_n*NCRIT+ntargets] = Precond[2,targets]
    Pre3Host[twig_n*NCRIT:twig_n*NCRIT+ntargets] = Precond[3,targets]

# Pack source parameters (position, area, vertices, etc.)
xsHost   = xj[srcPtr[0:offSrc]]
ysHost   = yj[srcPtr[0:offSrc]]
zsHost   = zj[srcPtr[0:offSrc]]
triHost  = srcPtr[0:offSrc]/K    # Triangle number
kHost    = srcPtr[0:offSrc]%K    # Gauss point number
AreaHost = Area[srcPtr[0:offSrc]]
normal_xHost = normal[srcPtr[0:offSrc],0]

vertexHost = zeros(offSrc*9)
for i in range(offSrc):
    vertexHost[i*9:i*9+9] = ravel(vertex[triangle[srcPtr[i]]])

# Pack multipole parameters (centers)
xcHost = zeros(offMlt)
ycHost = zeros(offMlt)
zcHost = zeros(offMlt)

i = -1
for C in mltPtr[0:offMlt]:
    i += 1
    xcHost[i] = Cells[C].xc
    ycHost[i] = Cells[C].yc
    zcHost[i] = Cells[C].zc
toc = time.time()
time_pack = toc - tic

### Transfer data to GPU
tic = cuda.Event()
toc = cuda.Event()

tic.record()
offsetSrcDev = gpuarray.to_gpu(offsetSrcHost.astype(int32))
offsetMltDev = gpuarray.to_gpu(offsetMltHost.astype(int32))
sizeTarDev   = gpuarray.to_gpu(sizeTarHost.astype(int32))
xsDev = gpuarray.to_gpu(xsHost.astype(REAL))
ysDev = gpuarray.to_gpu(ysHost.astype(REAL))
zsDev = gpuarray.to_gpu(zsHost.astype(REAL))
xcDev = gpuarray.to_gpu(xcHost.astype(REAL))
ycDev = gpuarray.to_gpu(ycHost.astype(REAL))
zcDev = gpuarray.to_gpu(zcHost.astype(REAL))
xtDev = gpuarray.to_gpu(xtHost.astype(REAL))
ytDev = gpuarray.to_gpu(ytHost.astype(REAL))
ztDev = gpuarray.to_gpu(ztHost.astype(REAL))
Pre0Dev = gpuarray.to_gpu(Pre0Host.astype(REAL))
Pre1Dev = gpuarray.to_gpu(Pre1Host.astype(REAL))
Pre2Dev = gpuarray.to_gpu(Pre2Host.astype(REAL))
Pre3Dev = gpuarray.to_gpu(Pre3Host.astype(REAL))
triDev = gpuarray.to_gpu(triHost.astype(int32))
kDev = gpuarray.to_gpu(kHost.astype(int32))
vertexDev = gpuarray.to_gpu(vertexHost.astype(REAL))
normal_xDev = gpuarray.to_gpu(normal_xHost.astype(REAL))
xkDev = gpuarray.to_gpu(xk.astype(REAL))
wkDev = gpuarray.to_gpu(wk.astype(REAL))
AreaDev = gpuarray.to_gpu(AreaHost.astype(REAL))

toc.record()
toc.synchronize()
time_trans = tic.time_till(toc)*1e-3

### Define CUDA block and grid sizes
BlocksPerTwig = int(ceil(NCRIT/float(BSZ))) 
print 'Block per twig: %i'%BlocksPerTwig

GSZ = int(ceil(float(Nround)/NCRIT))


tic = time.time()

### Solve system
phi = zeros(2*N)
phi = gmres_solver(Precond, E_hat, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev,normal_xDev, xj, yj, zj, xi, yi, zi, 
                    xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                    xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, 
                    sizeTarDev, offsetMltDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                    tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost,
                    offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, K, threshold, BSZ, GSZ, BlocksPerTwig, phi, F, 
                    restart, tol, max_iter, Cells, theta, Nm, II, JJ, KK, index, combII, combJJ, combKK, 
                    IImii, JJmjj, KKmkk, index_small, P, kappa, NCRIT, twig, eps, time_pack) 

toc = time.time()
solve_time = toc-tic

print 'Solve time        : %fs'%solve_time

### Calcualte reaction potential on charge positions
phi_reac = zeros(Nq)
for i in range(Nq):
    phi_reac[i] = sum(Area*phi[N:2*N]/R_pq[i] + Area*phi[0:N]/R_pq[i]**3*(dx_pq[i]*normal[:,0]+dy_pq[i]*normal[:,1]+dz_pq[i]*normal[:,2]))/(4*pi)

### Calculate solvation energy
JtoCal = 4.184
E_solv = 0.5*qe**2*sum(q*phi_reac)*Na*1e-3*1e10/(JtoCal*E_0)

### Calculate analytic solution
K_sph = 30 			# Number of terms in spherical harmonic expansion
a = R  				# Ion-exclusion layer radius
phi_P = an_P(q, xq, E_1, E_2, E_0, R, kappa, a, K_sph)
E_P = 0.5*qe**2*sum(q*phi_P)*Na*1e7/JtoCal

print '\n E_solv = %s, Legendre polynomial sol = %f, Error: %s'%(E_solv, E_P, abs(E_solv-E_P)/abs(E_P))
