#!/usr/bin/env python
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

from numpy 			import *
from math  			import pi
from scipy.misc     import factorial
import argparse
import time

# Import self made modules
import sys 
from gmres			    import gmres_solver, gmres_sigma
from projection         import get_phir
from classes            import surfaces, timings, parameters, index_constant, fill_surface, initializeSurf, initializeField, dataTransfer, fill_phi, computePrecond
from output             import printSummary
from matrixfree         import generateRHS, generateRHS_gpu, calculateEsolv, coulombEnergy, calculateEsurf, selfExterior, selfInterior, computeNormalElectricField

sys.path.append('../util')
from readData        import readVertex, readTriangle, readpqr, readParameters
from triangulation 	 import *
from an_solution     import an_P, two_sphere
from semi_analytical import *

sys.path.append('tree')
from FMMutils       import *
from cuda_kernels   import kernels

# import modules for testing
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

### Parse command line arguments
parser = argparse.ArgumentParser(description='PyGBe: Python GPU code for boundary elements')
parser.add_argument('parameter_file', help='Parameter file, see input_files/file.param for format details')
parser.add_argument('config_file', help='Configuration file, see input_files/file.config for format details')
parser.add_argument('--asymmetric', help='Activates nonlinear BCs and Picard iteration to consider asymmetric charging energy', action='store_true')
parser.add_argument('--chargeForm', help='Use apparent surface charge to compute the normal electric field', action='store_true')

args = parser.parse_args()

### Time stamp
timestamp = time.localtime()
print 'Run started on:'
print '\tDate: %i/%i/%i'%(timestamp.tm_year,timestamp.tm_mon,timestamp.tm_mday)
print '\tTime: %i:%i:%i'%(timestamp.tm_hour,timestamp.tm_min,timestamp.tm_sec)

TIC = time.time()
### Read parameters
param = parameters()
precision = readParameters(param, args.parameter_file)
configFile = args.config_file

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

'''
fig = plt.figure()
ax = Axes3D(fig)
#ss=surf_array[0]
for i in range(1):
    ss = surf_array[i]
    ax.scatter(ss.xi,ss.yi,ss.zi,c='b',marker='o')
    ax.scatter(ss.xi+ss.normal[:,0], ss.yi+ss.normal[:,1], ss.zi+ss.normal[:,2],c='r', marker='o')
plt.show()
quit()
'''

### Output setup summary
param.N   = 0
param.Neq = 0
for s in surf_array:
    N_aux = len(s.triangle)
    param.N += N_aux
    if s.surf_type == 'dirichlet_surface' or s.surf_type == 'neumann_surface' or s.surf_type == 'asc_surface':
        param.Neq += N_aux
    else:
        param.Neq += 2*N_aux
print '\nTotal elements : %i'%param.N
print 'Total equations: %i'%param.Neq

printSummary(surf_array, field_array, param)

### Precomputation
ind0 = index_constant()
computeIndices(param.P, ind0)
precomputeTerms(param.P, ind0)

### Load CUDA code
kernel = kernels(param.BSZ, param.Nm, param.K_fine, param.P, precision)

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
    dataTransfer(surf_array, field_array, ind0, param, kernel)
toc = time.time()
transfer_time = toc-tic

timing = timings()

### Generate RHS
print 'Generate RHS'
tic = time.time()
if param.GPU==0:
    F = generateRHS(field_array, surf_array, param, kernel, timing, ind0)
elif param.GPU==1:
    F = generateRHS_gpu(field_array, surf_array, param, kernel, timing, ind0)
toc = time.time()
rhs_time = toc-tic

savetxt('RHS.txt',F)

setup_time = toc-TIC
print 'List time          : %fs'%list_time
print 'Data transfer time : %fs'%transfer_time
print 'RHS generation time: %fs'%rhs_time
print '------------------------------'
print 'Total setup time   : %fs\n'%setup_time

tic = time.time()

### Solve
print 'Solve'
if args.asymmetric == True:
    Npicard = 10
else:
    Npicard = 1

alpha = 0.5
beta = -60
gamma = -0.5
mu = -alpha*tanh(-gamma)
phi = zeros(param.Neq)
for picardIter in range(Npicard):

    if args.asymmetric == True and args.chargeForm == False:
        print '\nPicard iteration %i'%picardIter
        for ss in range(len(surf_array)):
            s = surf_array[ss]
            if s.surf_type == 'dielectric_interface':

                if picardIter==0:
                    for ff in field_array:
                        if len(ff.parent)>0:
                            if ff.parent[0]==ss:
                                child_field = ff

                    s.dphi = -computeNormalElectricField(s, ss, child_field, param, zeros(len(s.triangle)), ind0, kernel, timing)

                h_nonlinear = alpha*tanh(-beta*s.dphi-gamma) + mu
                f_nonlinear = s.Ein/(s.Eout-s.Ein) - h_nonlinear
                s.E_hat = f_nonlinear/(1+f_nonlinear)
            if s.surf_type == 'stern_layer':
                qtotal = 0
                for f in field_array:
                    if len(f.q)>0:
                        qtotal += sum(f.q)
                sigma_total = sum(s.dphi*s.Area)
                if abs(sigma_total)<1e-10:
                    sigma_total = -qtotal/s.Eout
                s.E_hat = -(qtotal/s.Eout)/sigma_total
            computePrecond(s)

    if args.asymmetric == True and args.chargeForm == True:
        print '\nPicard iteration %i'%picardIter
        for ss in range(len(surf_array)):
            s = surf_array[ss]
            if s.surf_type == 'dielectric_interface':
                s.XinV = s.dphi 
                s.XinK = s.phi
                s.E_hat = 1.
                RHS_sigma, t1, t2 = selfExterior(s, ss, 1, param, ind0, timing, kernel)
                if picardIter==0: 
                    sigma = zeros(len(RHS_sigma))

                if sum(abs(RHS_sigma))>1e-12:
                    timing.AI_int = 0
                    sigma = gmres_sigma(s, ss, sigma, RHS_sigma, param, ind0, timing, kernel)
                for ff in field_array:
                    if len(ff.parent)>0:
                        if ff.parent[0]==ss:
                            child_field = ff

                ElecField = computeNormalElectricField(s, ss, child_field, param, sigma, ind0, kernel, timing) 

                h_nonlinear = alpha*tanh(beta*ElecField-gamma) + mu
                f_nonlinear = s.Ein/(s.Eout-s.Ein) - h_nonlinear
                s.E_hat = f_nonlinear/(1+f_nonlinear)
            if s.surf_type == 'stern_layer':
                qtotal = 0
                for f in field_array:
                    if len(f.q)>0:
                        qtotal += sum(f.q)
                sigma_total = sum(s.dphi*s.Area)
                if abs(sigma_total)<1e-10:
                    sigma_total = -qtotal/s.Eout
                s.E_hat = -(qtotal/s.Eout)/sigma_total
            computePrecond(s)


    phi = gmres_solver(surf_array, field_array, phi, F, param, ind0, timing, kernel) 

#   Put result phi in corresponding surfaces
    fill_phi(phi, surf_array)
    timing.AI_int = 0

toc = time.time()
solve_time = toc-tic
print 'Solve time        : %fs'%solve_time
savetxt('phi.txt',phi)
#phi = loadtxt('phi.txt')


### Calculate solvation energy
print '\nCalculate Esolv'
tic = time.time()
E_solv = calculateEsolv(surf_array, field_array, param, kernel)
toc = time.time()
print 'Time Esolv: %fs'%(toc-tic)
ii = -1
for f in param.E_field:
    parent_type = surf_array[field_array[f].parent[0]].surf_type
    if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface':
        ii += 1
        print 'Region %i: Esolv = %f kcal/mol = %f kJ/mol'%(f, E_solv[ii], E_solv[ii]*4.184)

### Calculate surface energy
print '\nCalculate Esurf'
tic = time.time()
E_surf = calculateEsurf(surf_array, field_array, param, kernel)
toc = time.time()
ii = -1
for f in param.E_field:
    parent_type = surf_array[field_array[f].parent[0]].surf_type
    if parent_type == 'dirichlet_surface' or parent_type == 'neumann_surface':
        ii += 1
        print 'Region %i: Esurf = %f kcal/mol = %f kJ/mol'%(f, E_surf[ii], E_surf[ii]*4.184)
print 'Time Esurf: %fs'%(toc-tic)

### Calculate Coulombic interaction
print '\nCalculate Ecoul'
tic = time.time()
i = -1
E_coul = []
for f in field_array:
    i += 1
    if f.coulomb == 1:
        print 'Calculate Coulomb energy for region %i'%i
        E_coul.append(coulombEnergy(f, param))
        print 'Region %i: Ecoul = %f kcal/mol = %f kJ/mol'%(i,E_coul[-1],E_coul[-1]*4.184)
toc = time.time()
print 'Time Ecoul: %fs'%(toc-tic)

### Output summary
print '\n--------------------------------'
print 'Totals:'
print 'Esolv = %f kcal/mol'%sum(E_solv)
print 'Esurf = %f kcal/mol'%sum(E_surf)
print 'Ecoul = %f kcal/mol'%sum(E_coul)
print '\nTime = %f s'%(toc-TIC)

# Analytic solution
'''
# two spheres
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
print '\n E_solv = %s kcal/mol, Analytical solution = %f kcal/mol, Error: %s'%(E_solv, E2an, abs(E_solv-E2an)/abs(E2an))
print '\n E_solv = %s kJ/mol, Analytical solution = %f kJ/mol, Error: %s'%(E_solv*JtoCal, E2an*JtoCal, abs(E_solv-E2an)/abs(E2an))

# sphere with stern layer
K_sph = 20 # Number of terms in spherical harmonic expansion
#E_1 = field_array[2].E # stern
E_1 = field_array[1].E # no stern
E_2 = field_array[0].E
R1 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0])
#R2 = norm(surf_array[1].vertex[surf_array[0].triangle[0]][0]) # stern
R2 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0]) # no stern
#q = field_array[2].q # stern
q = field_array[1].q # no stern
#xq = field_array[2].xq # stern
xq = field_array[1].xq # no stern
xq += 1e-12
print q, xq, E_1, E_2, R1, field_array[0].kappa, R2, K_sph
phi_P = an_P(q, xq, E_1, E_2, R1, field_array[0].kappa, R2, K_sph)
JtoCal = 4.184
E_P = 0.5*param.qe**2*sum(q*phi_P)*param.Na*1e7/JtoCal
print '\n E_solv = %s, Legendre polynomial sol = %f, Error: %s'%(E_solv, E_P, abs(E_solv-E_P)/abs(E_P))
'''
