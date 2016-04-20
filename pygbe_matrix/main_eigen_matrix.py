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

# This code solves the multisurface BEM for proteins  
# interacting with charged surfaces

from numpy.linalg   import eig, inv
from math           import pi
from scipy.misc     import factorial
from scipy.sparse.linalg   import gmres
from scipy.sparse   import *
import time

# Import self made modules
import sys 
sys.path.append('../util')
#from semi_analyticalwrap import SA_wrap_arr
from an_solution        import *
from integral_matfree   import *
from triangulation      import *
from class_definition   import surfaces, parameters, readParameters, initializeField, initializeSurf
from gmres              import gmres_solver
from blockMatrixGen     import blockMatrix, generateMatrix, generatePreconditioner
from RHScalculation     import charge2surf, generateRHS
from interactionCalculation import computeInter
from energyCalculation      import fill_phi, solvationEnergy, coulombicEnergy, surfaceEnergy

tic = time.time()
config_file = 'matrix_tests/input_files/sphere_single.config'
param_file ='matrix_tests/input_files/sphere.param'

print 'Parameters file: ' + param_file 
print 'Config file    : ' + config_file 

param = parameters()
readParameters(param, param_file)

field_array = initializeField(config_file, param)
surf_array, Neq  = initializeSurf(field_array, param, config_file)

i = -1
for f in field_array:
    i += 1
    print '\nField %i:'%i
    if f.LorY==1: 
        print 'Is a Laplace region'
    elif f.LorY==2:
        print 'Is a Yukawa region'
    else:
        print 'Is enclosed by a dirichlet or neumann surface'
    if len(f.parent)>0:
        print 'Is enclosed by surface %i'%(f.parent[0])
    else:
        print 'Is the solvent'
    if len(f.child)>0:
        print 'Contains surfaces ' + str(f.child)
    else:
        print 'Is an inner-most region'
    print 'Parameters: kappa: %f, E: %f'%(f.kappa, f.E)

print '\nTotal elements : %i'%param.N
print 'Total equations: %i'%param.Neq
    

JtoCal = 4.184

#### Compute interactions
print '\nCompute interactions'
computeInter(surf_array, field_array, param)

#### Generate RHS
print '\nGenerate RHS'
F, F_sym, X_sym, Nblock = generateRHS(surf_array, field_array, Neq)

print '\nRHS generated...'


#### Generate matrix
M, M_sym = generateMatrix(surf_array, Neq) 

print '\nSymbolic system:\n'
counter = 0
for i in range(len(M_sym)):
    for j in range(len(M_sym[i])):
        counter += 1
        buff = ''
        for k in range(len(M_sym[i][j])):
            for l in range(len(M_sym[i][j][k])):
                buff += M_sym[i][j][k][l]
        if counter==Nblock/2+1:
            print '|'+buff+'|  X  |'+X_sym[i][j]+'|  =  |'+F_sym[i][j]+'|'
        else:
            print '|'+buff+'|     |'+X_sym[i][j]+'|     |'+F_sym[i][j]+'|'

M *= 1/(4*pi)
Nh = len(surf_array[0].xi)
KL = M[0:Nh,0:Nh]
VL = -M[0:Nh,Nh:2*Nh]
KY = M[Nh:2*Nh,0:Nh]
VY = M[Nh:2*Nh,Nh:2*Nh]/surf_array[0].Ehat
for i in range(len(KL)):
    KL[i,i] = 0.5
    KY[i,i] = 0.5

KY_inv = inv(KY)
VL_inv = inv(VL)

print 'Matrices inverted'

M_aux1 = dot(VL_inv,KL)
M_aux2 = dot(VY,M_aux1)
Mat = dot(KY_inv,M_aux2)

print 'Matrix generated'

eVal,eVec = eig(-Mat)

for i in range(len(KL)):
    KL[i,i] = 0.

eVal2,eVec2 = eig(-KL)

Lambda = 2*real(eVal2)
E1_lap = 1*(1+Lambda)/(Lambda-1)
E1_yuk = 1/real(eVal)
w_lap = sqrt(1/(1-E1_lap))
w_yuk = sqrt(1/(1-E1_yuk))
print 'Yukawa-Laplace difference'
print abs(w_lap-w_yuk)[0:50]/abs(w_lap)[0:50]

P = zeros((len(M),3))
nx = surf_array[0].normal[:,0]
ny = surf_array[0].normal[:,1]
nz = surf_array[0].normal[:,2]
eVec_r = real(eVec)
for i in range(len(Mat)):
    P[i,0] = sum(eVec_r[:,i]*nx*surf_array[0].Area)
    P[i,1] = sum(eVec_r[:,i]*ny*surf_array[0].Area)
    P[i,2] = sum(eVec_r[:,i]*nz*surf_array[0].Area)

Plarge = where(sum(abs(P[0:50,:]),axis=1)>1e-10)[0]
test = 2*pi/real(eVal)
print 'Plarge'
print Plarge
print abs(w_lap-w_yuk)[Plarge]/abs(w_lap)[Plarge] 

NN = len(surf_array[0].xi)
X = zeros((NN,3))
X[:,0] = surf_array[0].xi
X[:,1] = surf_array[0].yi
X[:,2] = surf_array[0].zi

savetxt('eVectest',real(eVec_r[:,0:30]))
savetxt('surface',X)
quit()

# Generate preconditioner
# Inverse of block diagonal matrix
print '\n\nGenerate preconditioner'
Ainv = generatePreconditioner(surf_array)

print 'preconditioner generated'

MM = Ainv*M
FF = Ainv*F
#MM = M
#FF = F

savetxt('RHS_matrix.txt',FF)

print '\nCalculation for both spheres'
tec = time.time()
phi = zeros(len(F))
phi = gmres_solver(MM, phi, FF, param.restart, param.tol, param.max_iter) # Potential both spheres
converged = -1
toc = time.time()

savetxt('phi_matrix.txt',phi)

print '\nEnergy calculation'
fill_phi(phi, surf_array)

Esolv, field_Esolv = solvationEnergy(surf_array, field_array, param)

Ecoul, field_Ecoul = coulombicEnergy(field_array, param)

Esurf, surf_Esurf = surfaceEnergy(surf_array, param)

toc = time.time()

print 'Esolv:'
for i in range(len(Esolv)):
    print 'Region %i: %f kcal/mol'%(field_Esolv[i],Esolv[i])
print 'Esurf:'
for i in range(len(Esurf)):
    print 'Surface %i: %f kcal/mol'%(surf_Esurf[i],Esurf[i])
print 'Ecoul:'
for i in range(len(Ecoul)):
    print 'Region %i: %f kcal/mol'%(field_Ecoul[i],Ecoul[i])

print '\nTotals:'
print 'Esolv = %f kcal/mol'%sum(Esolv)
print 'Esurf = %f kcal/mol'%sum(Esurf)
print 'Ecoul = %f kcal/mol'%sum(Ecoul)
print '\nTime = %f s'%(toc-tic)
