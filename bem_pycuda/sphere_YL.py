"""
Spherical molecule with random charge distribution
Yoon and Lenhoff 1990 formulation

Analytical solution:
Solvation Energy = -39.4326 kcal/mol
"""

from pylab 			import *
from math  			import pi
from scipy.misc     import factorial
import time

# Import self made modules
import sys 
from gmres			    import gmres_solver    
from projection         import get_phir
from classes            import surfaces, timings, parameters, index_constant, fill_surface, initializeSurf, initializeField, dataTransfer
from output             import printSummary
from matrixfree         import generateRHS, generateRHS_gpu, calculateEsolv

sys.path.append('../util')
from readData        import readVertex, readTriangle, readpqr, readParameters
from triangulation 	 import *
from an_solution     import an_P, two_sphere
from semi_analytical import *

sys.path.append('tree')
from FMMutils       import *
from cuda_kernels   import kernels

### Read parameters
param = parameters()
precision = readParameters(param,'input_files/parameters_sphere.txt')
configFile = 'input_files/config_sphere_stern.txt'

# Derived parameters 
param.Nm            = (param.P+1)*(param.P+2)*(param.P+3)/6     # Number of terms in Taylor expansion
param.BlocksPerTwig = int(ceil(param.NCRIT/float(param.BSZ)))   # CUDA blocks that fit per twig

### Generate array of fields
field_array = initializeField(configFile, param)

### Generate array of surfaces and read in elements
surf_array = initializeSurf(field_array, configFile, param)

### Fill surface class
time_sort = 0.
for i in range(len(surf_array)):
    time_sort += fill_surface(surf_array[i], param)

### Output setup summary
param.N = 0
for i in range(len(surf_array)):
    N_aux = len(surf_array[i].triangle)
    param.N += N_aux
print 'Total elements: %i'%param.N
printSummary(surf_array, field_array, param)

tac = time.time()

### Precomputation
ind0 = index_constant()
computeIndices(param.P, ind0)
precomputeTerms(param.P, ind0)

### Load CUDA code
kernel = kernels(param.BSZ, param.Nm, param.Nk, param.P, precision)

### Generate interaction list
print 'Generate interaction list'
tic = time.time()
generateList(surf_array, field_array, param)
toc = time.time()
list_time = toc-tic

### Transfer data to GPU
print 'Transfer data to GPU'
tic = time.time()
if param.GPU==1:
    dataTransfer(surf_array, field_array, ind0, param)
toc = time.time()
transfer_time = toc-tic

### Generate RHS
print 'Generate RHS'
tic = time.time()
if param.GPU==0:
    F = generateRHS(field_array, surf_array, param.N)
elif param.GPU==1:
    F = generateRHS_gpu(field_array, surf_array, param, kernel)
toc = time.time()
rhs_time = toc-tic

setup_time = toc-tac
print 'List time          : %fs'%list_time
print 'Data transfer time : %fs'%transfer_time
print 'RHS generation time: %fs'%rhs_time
print '------------------------------'
print 'Total setup time   : %fs\n'%setup_time

tic = time.time()

### Solve
print 'Solve'
timing = timings()
phi = zeros(2*param.N)
phi = gmres_solver(surf_array, field_array, phi, F, param, ind0, timing, kernel) 
toc = time.time()
solve_time = toc-tic
print 'Solve time        : %fs'%solve_time

savetxt('phi.txt',phi)
#phi = loadtxt('phi.txt')

### Calculate solvation energy
print 'Calculate Esolv'
tic = time.time()
E_solv = calculateEsolv(phi, surf_array, field_array, param, kernel)
toc = time.time()
print 'Time Esolv: %f'%(toc-tic)
print 'Esolv: %f'%E_solv

# Analytic solution
# two spheres
'''
R1 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0])
dist = norm(field_array[2].xq[0]-field_array[1].xq[0])
E_1 = field_array[1].E
E_2 = field_array[0].E
E_an,E1an,E2an = two_sphere(R1, dist, field_array[0].kappa, E_1, E_2, field_array[1].q[0])
JtoCal = 4.184
C0 = param.qe**2*param.Na*1e-3*1e10/(JtoCal*param.E_0)
E_an *= C0/(4*pi)
E1an *= C0/(4*pi)
E2an *= C0/(4*pi)
print '\n E_solv = %s, Analytical solution = %f, Error: %s'%(E_solv, E2an, abs(E_solv-E2an)/abs(E2an))
'''

# sphere with stern layer
K_sph = 10 # Number of terms in spherical harmonic expansion
E_1 = field_array[2].E # stern
#E_1 = field_array[1].E # no stern
E_2 = field_array[0].E
R1 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0])
R2 = norm(surf_array[1].vertex[surf_array[0].triangle[0]][0]) # stern
#R2 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0]) # no stern
q = field_array[2].q # stern
#q = field_array[1].q # no stern
xq = field_array[2].xq # stern
#xq = field_array[1].xq # no stern
phi_P = an_P(q, xq, E_1, E_2, param.E_0, R1, field_array[0].kappa, R2, K_sph)
JtoCal = 4.184
E_P = 0.5*param.qe**2*sum(q*phi_P)*param.Na*1e7/JtoCal
print '\n E_solv = %s, Legendre polynomial sol = %f, Error: %s'%(E_solv, E_P, abs(E_solv-E_P)/abs(E_P))

