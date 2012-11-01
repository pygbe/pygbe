"""
Spherical molecule with random charge distribution
Yoon and Lenhoff 1990 formulation

Analytical solution:
Solvation Energy = -39.4326 kcal/mol
"""

from pylab 			import *
from math  			import pi
from scipy.misc     import factorial
from numpy          import float64 as REAL
import time

# Import self made modules
import sys 
from gmres			    import gmres_solver    
from matrixfree         import getGaussPoints, get_phir, get_phir_gpu

sys.path.append('../geometry')
from readData import readVertex, readTriangle, readpqr

sys.path.append('../util')
from triangulation 	 import *
from an_solution     import an_P
from integral_matfree import *
from semi_analytical import *

sys.path.append('tree')
from FMMutils import *

# pyCUDA libraries
import pycuda.autoinit
import pycuda.gpuarray  as gpuarray
import pycuda.driver    as cuda
from cuda_kernels import kernels



tic = time.time()
Rec = 7		# Number of recursions to generate sphere
R   = 4. 	# Radius of sphere
K   = 1  	# Number of gauss points

threshold = 0.8 	# L/d threshold to change to analytical
                    # Over: analytical, under: quadrature

# Physical variables
qe = 1.60217646e-19             # Electron charge 
#Nq = 1                          # Number of charges
#q  = array([1.])                # Charge in electron charge units
#xq = array([[2e0,-1e-10,0.]])  # Locations of charges
#xq = array([[0.5,sqrt(1.75),sqrt(2)]])  # Locations of charges
#xq = array([[1,1,sqrt(2)]])  # Locations of charges
Na = 6.0221415e23               # Avogadro's number
E_0 = 8.854187818e-12           # Vacuum dielectric constant
E_1 = 4.                        # Molecule dielectric constant
E_2 = 80.                       # Water dielectric constant
E_hat = REAL(E_1/E_2)
kappa = 1e-12#0.125                   # Screening parameter


### GMRES related variables
restart  = 200 	# Restart for GMRES
tol		 = REAL(1e-4) # tolerance of GMRES
max_iter = 1000 # maximum number of iterations for GMRES

### Tree code related variables
P     = 2                                         # Order of Taylor expansion
eps   = 1e-10                                     # Epsilon machine
Nm    = int32(factorial(P+3)/(6*factorial(P)))    # Number of terms in Taylor expansion
NCRIT = 500                                       # Max number of particles per twig cell
theta = 0.6                                       # MAC criterion
BSZ   = 128                                       # CUDA block size

### Read in charges
xq,q,Nq = readpqr('../geometry/single/built_parse.crd')
rq = sqrt(xq[:,0]**2+xq[:,1]**2+xq[:,2]**2)


Area_null = []

### Setup elements
vertex_x,vertex_y, vertex_z = readVertex('../geometry/single/surfwQDCVE.vert')
triangle_raw = readTriangle('../geometry/single/surfwQDCVE.face')
#vertex_x,vertex_y, vertex_z = readVertex('../geometry/huge/surfnHIAnp.vert')
#triangle_raw = readTriangle('../geometry/huge/surfnHIAnp.face')
#vertex_x,vertex_y, vertex_z = readVertex('../geometry/huge/surf8DPl7i.vert')
#triangle_raw = readTriangle('../geometry/huge/surf8DPl7i.face')
#vertex_x,vertex_y, vertex_z = readVertex('../geometry/sphere_r5.vert')
#triangle_raw = readTriangle('../geometry/sphere_r5.face')

vertex = zeros((len(vertex_x),3))
vertex[:,0] = vertex_x.astype(REAL)
vertex[:,1] = vertex_y.astype(REAL)
vertex[:,2] = vertex_z.astype(REAL)

for i in range(len(triangle_raw)):
    L0 = vertex[triangle_raw[i,1]] - vertex[triangle_raw[i,0]]
    L2 = vertex[triangle_raw[i,0]] - vertex[triangle_raw[i,2]]
    normal_aux = cross(L0,L2)
    Area_aux = linalg.norm(normal_aux)/2
    if Area_aux<1e-10:
        Area_null.append(i)

triangle = delete(triangle_raw, Area_null, 0)
'''

vertex, triangle, center = create_unit_sphere(Rec)
vertex *= R
center *= R
'''
toc = time.time()
time_read = toc - tic

tac = time.time()

N 		= len(triangle) # Number of elements (and targets)
Nj      = N*K           # Number of gauss points (sources)
bc     	= zeros(N)

print("**** "+str(N)+" elements ****")
print 'Number of charges: %i'%Nq
print 'Removed areas=0  : %i'%len(Area_null)
print 'Read file time: %fs'%time_read
print 'P: %i'%P

normal = zeros((N,3), dtype=REAL)
Area = zeros(N, dtype=REAL)
for i in range(N):
    L0 = vertex[triangle[i,1]] - vertex[triangle[i,0]]
    L2 = vertex[triangle[i,0]] - vertex[triangle[i,2]]
    normal_aux  = cross(L0,L2)
    Area_aux    = linalg.norm(normal_aux)/2
    Area[i]     = Area_aux
    normal[i,:] = normal_aux/(2*Area_aux)

print 'Elements per sq angs: %f'%(1./average(Area))

xi = zeros(N, dtype=REAL)
yi = zeros(N, dtype=REAL)
zi = zeros(N, dtype=REAL)
for i in range(N):
    xi[i] = average(vertex[triangle[i],:,0])
    yi[i] = average(vertex[triangle[i],:,1])
    zi[i] = average(vertex[triangle[i],:,2])


# Set Gauss points (sources)
xj,yj,zj = getGaussPoints(vertex,triangle,K)

x_center = zeros(3)
x_center[0] = average(xi).astype(REAL)
x_center[1] = average(yi).astype(REAL)
x_center[2] = average(zi).astype(REAL)
dist = sqrt((xi-x_center[0])**2+(yi-x_center[1])**2+(zi-x_center[2])**2)
R_C0 = max(dist) + 1.
print 'C0 cell size: %f'%R_C0
print 'C0 cell pos: %f, %f, %f'%(x_center[0], x_center[1], x_center[2])


Nk = 5          # Number of Gauss points per side for semi-analytical integrals
xk,wk = GQ_1D(Nk)
xk = REAL(xk)
wk = REAL(wk)

# Generate tree, compute indices and precompute terms for M2M
Cells = generateTree(xi,yi,zi,NCRIT,Nm,N,R_C0,x_center)
twig = []
C = 0
twig = findTwigs(Cells, C, twig, NCRIT)

II,JJ,KK,index, index_large = computeIndices(P)
addSources(Cells,twig,K)

combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, index_ptr = precompute_terms(P, II, JJ, KK)

# Generate preconditioner
# Will use block-diagonal preconditioner (AltmanBardhanWhiteTidor2008)
dX11 = zeros(N, dtype=REAL) # Top left block
dX12 = zeros(N, dtype=REAL) # Top right block
dX21 = zeros(N, dtype=REAL) # Bottom left block
dX22 = zeros(N, dtype=REAL) # Bottom right block
Precond = zeros((4,N), dtype=REAL)  # Stores the inverse of the block diagonal (also a tridiag matrix)
                        # Order: Top left, top right, bott left, bott right    


for i in range(N):
    panel = vertex[triangle[i]]
    center = array([xi[i],yi[i],zi[i]])
    same = array([1], dtype=int32)
    Aaux = zeros(1, dtype=REAL) # Top left
    Baux = zeros(1, dtype=REAL) # Top right
    Caux = zeros(1, dtype=REAL) # Bottom left
    Daux = zeros(1, dtype=REAL) # Bottom right

    SA_wrap_arr(ravel(panel), center, Daux, Caux, Baux, Aaux, kappa, same, xk, wk)
    dX11[i] = -Aaux
    dX12[i] = Baux
    dX21[i] = Caux
    dX22[i] = -E_hat*Daux


d_aux = 1/(dX22-dX21*dX12/dX11)
Precond[0,:] = 1/dX11 + 1/dX11*dX12*d_aux*dX21/dX11
Precond[1,:] = -1/dX11*dX12*d_aux
Precond[2,:] = -d_aux*dX21/dX11
Precond[3,:] = d_aux

#test = -0.0075788068139*ones(N)
#print sum(test-Precond[0,:])
#quit()

#print Precond[0,:]
#print Precond[1,:]
#print Precond[2,:]
#print Precond[3,:]
#quit()

rr = zeros(len(Cells))
for i in range(len(Cells)):
    rr[i] = Cells[i].r
Levels = log(Cells[0].r/min(rr))/log(2) + 1

print 'Cells : %i'%len(Cells)
print 'Twigs : %i'%len(twig)
print 'Levels: %i'%Levels
print 'Twig cell size   : %f'%(min(rr))
print 'Rbox/theta       : %f'%(min(rr)/theta)
print 'Analytic distance: %f\n'%(average(sqrt(2*Area))/threshold)

# Generate RHS
tic = time.time()

'''
R_pq = zeros((Nq,N), dtype=REAL)
for i in range(Nq):
    dx_pq = xq[i,0] - xi 
    dy_pq = xq[i,1] - yi 
    dz_pq = xq[i,2] - zi 
    R_pq[i,:]  = sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)

R_pq  = sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)
F = zeros(2*N)
F[0:N] = sum(-transpose(q*ones((N,Nq)))/(E_1*R_pq),axis=0) * Precond[0,:]
F[N:2*N] = sum(-transpose(q*ones((N,Nq)))/(E_1*R_pq),axis=0) * Precond[2,:]
del R_pq
'''

F_gpu = gpuarray.zeros(2*N, dtype=REAL)
xq_aux = zeros(Nq, dtype=REAL)
xq_aux[:] = xq[:,0]
xq_gpu = cuda.to_device(xq_aux)
xq_aux[:] = xq[:,1]
yq_gpu = cuda.to_device(xq_aux)
xq_aux[:] = xq[:,2]
zq_gpu = cuda.to_device(xq_aux)
xi_gpu = cuda.to_device(xi)
yi_gpu = cuda.to_device(yi)
zi_gpu = cuda.to_device(zi)
q_gpu  = cuda.to_device(q)
P0_gpu = cuda.to_device(Precond[0,:]) 
P2_gpu = cuda.to_device(Precond[2,:]) 

GSZ = int(ceil(float(N)/BSZ))
mod = kernels(BSZ,Nm,Nk,P)
compute_RHS_gpu = mod.get_function("compute_RHS")
compute_RHS_gpu(F_gpu, xq_gpu, yq_gpu, zq_gpu, 
                q_gpu, xi_gpu, yi_gpu, zi_gpu, 
                P0_gpu, P2_gpu, int32(N), int32(Nq), REAL(E_1), 
                block=(BSZ,1,1), grid=(GSZ,1))
F = zeros(2*N)
F_gpu.get(F)

#savetxt('RHS.txt',F)
#F = loadtxt('RHSCPU.txt')

toc = time.time()
rhs_time = toc-tic

# Data packing
tic = time.time()
Nround = len(twig)*NCRIT
Nlist  = Nround*len(twig)

offsetTarHost = zeros(len(twig)  , dtype=int32)
sizeTarHost   = zeros(len(twig)  , dtype=int32)
offsetSrcHost = zeros(len(twig)+1, dtype=int32)
offsetMltHost = zeros(len(twig)+1, dtype=int32)
offsetIntHost = zeros(len(twig)+1, dtype=int32)

tarPtr = zeros(Nround, dtype=int32)
srcPtr = zeros(Nj , dtype=int32)
mltPtr = zeros(len(twig)*len(twig), dtype=int32)
intPtr = zeros(len(twig)*len(twig), dtype=int32)

t = -1
offSrc = 0 
offMlt = 0 
offInt = 0 
for CI in twig:
    t += 1
    offsetSrcHost[t] = offSrc
    offsetMltHost[t] = offMlt
    offsetIntHost[t] = offInt
    CJ = 0 
    intPtr, mltPtr, offInt, offMlt = packData(Cells, CJ, CI, intPtr, mltPtr, offInt, offMlt, theta, NCRIT)
    srcPtr[offSrc:offSrc+Cells[CI].nsource] = Cells[CI].source
    offSrc += Cells[CI].nsource
offsetIntHost[-1] = offInt
offsetSrcHost[-1] = offSrc
offsetMltHost[-1] = offMlt

intPtrHost = zeros(offInt, dtype=int32)
for i in range(len(twig)):
    twig_index = where(intPtr[0:offInt]==twig[i])[0]
    intPtrHost[twig_index] = i

xtHost = zeros(Nround, dtype=REAL)
ytHost = zeros(Nround, dtype=REAL)
ztHost = zeros(Nround, dtype=REAL)
Pre0Host = zeros(Nround, dtype=REAL)
Pre1Host = zeros(Nround, dtype=REAL)
Pre2Host = zeros(Nround, dtype=REAL)
Pre3Host = zeros(Nround, dtype=REAL)

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

print 'size of packed source arrays: %i'%offSrc
print 'size of packed multip arrays: %i'%offMlt
print 'size of packed target arrays: %i\n'%len(xtHost)

xsHost   = xj[srcPtr[0:offSrc]]
ysHost   = yj[srcPtr[0:offSrc]]
zsHost   = zj[srcPtr[0:offSrc]]
triHost  = srcPtr[0:offSrc]/K    # Triangle number
kHost    = srcPtr[0:offSrc]%K    # Gauss point number
AreaHost = Area[triHost]
normal_xHost = 0#normal[triHost,0]
normal_yHost = 0#normal[triHost,1]
normal_zHost = 0#normal[triHost,2]

vertexHost = zeros(offSrc*9)
for i in range(offSrc):
    vertexHost[i*9:i*9+9] = ravel(vertex[triangle[triHost[i]]])

xcHost = zeros(offMlt, dtype=REAL)
ycHost = zeros(offMlt, dtype=REAL)
zcHost = zeros(offMlt, dtype=REAL)


i = -1
for C in mltPtr[0:offMlt]:
    i += 1
    xcHost[i] = Cells[C].xc
    ycHost[i] = Cells[C].yc
    zcHost[i] = Cells[C].zc
toc = time.time()
time_pack = toc - tic

# Transfer data to GPU
tic = cuda.Event()
toc = cuda.Event()

tic.record()
offsetIntDev = gpuarray.to_gpu(offsetIntHost.astype(int32))
intPtrDev = gpuarray.to_gpu(intPtrHost.astype(int32))
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
triDev = 0.#gpuarray.to_gpu(triHost.astype(int32))
kDev = gpuarray.to_gpu(kHost.astype(int32))
vertexDev = gpuarray.to_gpu(vertexHost.astype(REAL))
normal_xDev = 0.#gpuarray.to_gpu(normal_xHost.astype(REAL))
normal_yDev = 0.#gpuarray.to_gpu(normal_yHost.astype(REAL))
normal_zDev = 0.#gpuarray.to_gpu(normal_zHost.astype(REAL))
xkDev = gpuarray.to_gpu(xk.astype(REAL))
wkDev = gpuarray.to_gpu(wk.astype(REAL))
AreaDev = gpuarray.to_gpu(AreaHost.astype(REAL))
IndexDev = gpuarray.to_gpu(index_large.astype(int32))

toc.record()
toc.synchronize()
time_trans = tic.time_till(toc)*1e-3

BlocksPerTwig = int(ceil(NCRIT/float(BSZ))) 
print 'Block per twig: %i\n'%BlocksPerTwig

GSZ = int(ceil(float(Nround)/NCRIT))

tec = time.time()
setup_time = tec - tac
print 'Compute RHS time  : %fs'%rhs_time
print 'Data packing time : %fs'%time_pack
print 'Data transfer time: %fs'%time_trans
print '-----------------------------'
print 'Total setup time  : %fs\n'%setup_time

tic = time.time()

phi = zeros(2*N)
phi = gmres_solver(Precond, E_hat, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, 
                    AreaHost, AreaDev,normal_xDev, normal_yDev, normal_zDev, xj, yj, zj, xi, yi, zi, 
                    xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                    xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, 
                    sizeTarDev, offsetMltDev, offsetIntDev, intPtrDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                    tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost,
                    offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, K, threshold, BSZ, GSZ, BlocksPerTwig, phi, F, 
                    restart, tol, max_iter, Cells, theta, Nm, II, JJ, KK, index, index_large, IndexDev, combII, combJJ, combKK, 
                    IImii, JJmjj, KKmkk, index_small, index_ptr, P, kappa, NCRIT, twig, eps) 

toc = time.time()
solve_time = toc-tic

print 'Solve time        : %fs'%solve_time
savetxt('phi.txt',phi)
#phi = loadtxt('phi.txt')

tic = time.time()

'''
phi_reac2 = zeros(Nq)
M  = Area/(R_pq*4*pi)
dM = Area/R_pq**3*(dx_pq*normal[:,0] + dy_pq*normal[:,1] + dz_pq*normal[:,2])/(4*pi) 
L = sqrt(2*Area)
an_condition = greater_equal(L/R_pq,0.1)

for i in range(Nq):
    an_integrals = nonzero(an_condition[i])[0]
    G, dG = AI_arr(2,vertex[triangle[an_integrals]], xq[i], zeros(len(an_integrals)), 1.)
    M [i,an_integrals] = G/(4*pi)
    dM[i,an_integrals] = dG/(4*pi)
#    phi_reac[i] = sum(G*phi[N:2*N]+dG*phi[0:N])/(4*pi)
phi_reac2 = sum(M*phi[N:2*N] + dM*phi[0:N], axis=1)
'''


print '\nCalculate E_solv'
threshold = 0.1
P = 5
theta = 0.0
Nm= int32(factorial(P+3)/(6*factorial(P))) 
Nk = 5          # Number of Gauss points per side for semi-analytical integrals
xk,wk = GQ_1D(Nk)
xk = REAL(xk)
wk = REAL(wk)
'''
for C in range(len(Cells)):
    Cells[C].M  = zeros(Nm)
    Cells[C].Mx = zeros(Nm)
    Cells[C].My = zeros(Nm)
    Cells[C].Mz = zeros(Nm)
II,JJ,KK,index = computeIndices(P)
combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small = precompute_terms(P, II, JJ, KK)

phi_reac = get_phir(phi, vertex, triangle, xj, yj, zj, xi, yi, zi, xq[:,0], xq[:,1], xq[:,2],
                    Area, normal, xk, wk, Cells, theta, Nm, II, JJ, KK, index, 
                    combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, 
                    P, kappa, NCRIT, K, threshold, eps)
'''
phi_reac = get_phir_gpu(phi, vertex, triangle, xj, yj, zj, 
                            xq[:,0], xq[:,1], xq[:,2], Area, normal, xk, wk, 
                            K, threshold, BSZ, mod)


phi_reac /= (4*pi)
#phierror = sqrt(sum((phi_reac2-phi_reac)**2)/sum(phi_reac2**2))
#print 'phierror %f'%phierror

JtoCal = 4.184
E_solv = 0.5*qe**2*sum(q*phi_reac)*Na*1e-3*1e10/(JtoCal*E_0)
#E_solv2 = 0.5*qe**2*sum(q*phi_reac2)*Na*1e-3*1e10/(JtoCal*E_0)

toc = time.time()

print 'E_solv calculation time: %fs'%(toc-tic)

print '\nE_solv = %s'%(E_solv)
#print '\nE_solv2 = %s'%(E_solv2)

# Analytic solution
'''
#an = -39.4383
K_sph = 10 # Number of terms in spherical harmonic expansion
a = R  # Radius of position where ions can get closest to molecule
phi_P = an_P(q, xq, E_1, E_2, E_0, R, kappa, a, K_sph)
E_P = 0.5*qe**2*sum(q*phi_P)*Na*1e7/JtoCal

print '\n E_solv = %s, Legendre polynomial sol = %f, Error: %s'%(E_solv, E_P, abs(E_solv-E_P)/abs(E_P))
'''
