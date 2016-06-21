"""
This code solves the multisurface BEM for proteins interacting with charged
surfaces. 

 -It returns the Solvation Energy, Surface Energy and Coulombic Energy. 

 -If we have an incident electric field, it returns the extinction cross
  section. 
"""
from numpy                  import pi
from scipy.misc             import factorial
from scipy.sparse.linalg    import gmres
from argparse               import ArgumentParser

from class_definition       import (surfaces, parameters, readParameters, 
                            initializeField, initializeSurf, readElectricField)
from gmres                  import gmres_mgs
from blockMatrixGen         import (blockMatrix, generateMatrix,
                            generatePreconditioner)
from RHScalculation         import charge2surf, generateRHS
from interactionCalculation import computeInter
from energyCalculation      import (fill_phi, solvationEnergy, coulombicEnergy,
                            surfaceEnergy, dipoleMoment, extCrossSection)
import time

def read_inputs():
    """
    Parse command-line arguments to run main_matrix.

    User should provide:
    -param : str, parameter file name.
    -config: str, config file name.
    """

    parser = ArgumentParser(description='Manage main_matrix command line arguments')

    parser.add_argument('-p', '--param', dest='param', type=str, default=None,
                        help="Path to problem param file")

    parser.add_argument('-c', '--config', dest='config', type=str, default=None,
                        help="Path to problem config file")
    
    return parser.parse_args()

args = read_inputs()

tic = time.time()
param_file  = args.p
config_file = args.c

print 'Parameters file: ' + param_file 
print 'Config file    : ' + config_file 

param = parameters()
readParameters(param, param_file)

field_array = initializeField(config_file, param)
surf_array, Neq  = initializeSurf(field_array, param, config_file)

electricField, wavelength = readElectricField(config_file)

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
    if type(f.E)==complex:
        print 'Parameters: kappa: %f, E: %f+%fj'%(f.kappa, f.E.real, f.E.imag)
    else:
        print 'Parameters: kappa: %f, E: %f'%(f.kappa, f.E)

print '\nTotal elements : %i'%param.N
print 'Total equations: %i'%param.Neq
    

JtoCal = 4.184

#### Compute interactions
print '\nCompute interactions'
computeInter(surf_array, field_array, param)

#### Generate RHS
print '\nGenerate RHS'
F, F_sym, X_sym, Nblock = generateRHS(surf_array, field_array, Neq, electricField)

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

# Generate preconditioner
# Inverse of block diagonal matrix
print '\n\nGenerate preconditioner'
Ainv = generatePreconditioner(surf_array)

print 'preconditioner generated'

MM = Ainv*M
FF = Ainv*F
#MM = M
#FF = F

if type(MM[0,0]) != numpy.complex128:
    savetxt('RHS_matrix.txt',FF)

print '\nSolve system'
tec = time.time()
phi = zeros(len(F))

if type(MM[0,0]) == numpy.complex128:
    phi = gmres(MM, FF, tol=param.tol, restart=param.restart, maxiter=param.max_iter)[0]
else:
    phi = gmres_solver(MM, phi, FF, param.restart, param.tol, param.max_iter) 

converged = -1
toc = time.time()

if type(MM[0,0]) != numpy.complex128:
    savetxt('phi_matrix.txt',phi)


print '\nEnergy calculation'
fill_phi(phi, surf_array)

Esolv, field_Esolv = solvationEnergy(surf_array, field_array, param)

Ecoul, field_Ecoul = coulombicEnergy(field_array, param)

Esurf, surf_Esurf = surfaceEnergy(surf_array, param)

dipoleMoment(surf_array, electricField)

if abs(electricField)>1e-12:
    Cext, surf_Cext = extCrossSection(surf_array, array([1,0,0]), array([0,0,1]), wavelength, electricField)

toc = time.time()


print 'Esolv:'
for i in range(len(Esolv)):
    if type(Esolv[i])!=numpy.complex128:
        print 'Region %i: %f kcal/mol'%(field_Esolv[i],Esolv[i])
    else:
        print 'Region %i: %f + %fj kcal/mol'%(field_Esolv[i],Esolv[i].real,Esolv[i].imag)

print '\nEsurf:'
for i in range(len(Esurf)):
    if type(Esurf[i])!=numpy.complex128:
        print 'Surface %i: %f kcal/mol'%(surf_Esurf[i],Esurf[i])
    else:
        print 'Surface %i: %f + %fj kcal/mol'%(surf_Esurf[i],Esurf[i].real,Esurf[i].imag)

print '\nEcoul:'
for i in range(len(Ecoul)):
    print 'Region %i: %f kcal/mol'%(field_Ecoul[i],Ecoul[i])

if abs(electricField)>1e-12:
    print '\nCext:'
    for i in range(len(Cext)):
        print 'Surface %i: %f nm^2'%(surf_Cext[i], Cext[i])

print '\nTotals:'
print 'Esolv = %f + %fj kcal/mol'%(sum(Esolv).real,sum(Esolv).imag)
print 'Esurf = %f + %fj kcal/mol'%(sum(Esurf).real,sum(Esurf).imag)
print 'Ecoul = %f kcal/mol'%sum(Ecoul)
print '\nTime = %f s'%(toc-tic)
