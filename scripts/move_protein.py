#!/usr/bin/env python
# Calculated according to FelderPriluskySilmanSussman2007, but using center of mass
import numpy
from math import atan2

import os
import sys
from argparse import ArgumentParser
from pygbe.util.readData import readVertex, readpqr

def findDipole(xq, q):

    ctr = numpy.sum(numpy.transpose(xq)*numpy.abs(q), axis=1)/numpy.sum(numpy.abs(q))
#    ctr = average(xq, axis=0)
    r = xq - ctr
    d = numpy.sum(numpy.transpose(r)*q, axis=1)

    return d

def rotate_x(x, angle):

    xnew = numpy.zeros(numpy.shape(x))
    xnew[:,0] = x[:,0]
    xnew[:,1] = x[:,1]*numpy.cos(angle) - x[:,2]*numpy.sin(angle)
    xnew[:,2] = x[:,1]*numpy.sin(angle) + x[:,2]*numpy.cos(angle)

    return xnew


def rotate_y(x, angle):

    xnew = numpy.zeros(numpy.shape(x))
    xnew[:,0] = x[:,2]*numpy.sin(angle) + x[:,0]*numpy.cos(angle)
    xnew[:,1] = x[:,1]
    xnew[:,2] = x[:,2]*numpy.cos(angle) - x[:,0]*numpy.sin(angle)

    return xnew

def rotate_z(x, angle):

    xnew = numpy.zeros(numpy.shape(x))
    xnew[:,0] = x[:,0]*numpy.cos(angle) - x[:,1]*numpy.sin(angle)
    xnew[:,1] = x[:,0]*numpy.sin(angle) + x[:,1]*numpy.cos(angle)
    xnew[:,2] = x[:,2]

    return xnew

def modifypqr(inpqr, outpqr, xq):

    file_o = open(outpqr,'w')
    for line in file(inpqr):
        line_split = line.split()

        if line_split[0] == 'ATOM':
            atm_nu = int(line_split[1])-1
            line_add = ' %3.3f  %3.3f  %3.3f '%(xq[atm_nu,0], xq[atm_nu,1], xq[atm_nu,2])
            line_new = line[:27] + line_add + line[55:]
            file_o.write(line_new)
        else:
            file_o.write(line)
    file_o.close()

def read_inputs():
    """
    Parse command-line arguments to move protein.
    User should provide:
    - Problem folder (can be inferred from files if not provided)
    - Mesh to be rotated
    - PQR file
    - Alpha Y (change in degrees in Y)
    - Alpha Z (change in degrees in Z)
    - Name to append to rotated mesh
    """

    parser = ArgumentParser(description='Manage move_protein command line arguments')
    parser.add_argument('problem_folder', type=str,
                        help="Path to folder containing problem files")
    parser.add_argument('-m', '--mesh', dest='mesh', type=str, default=None,
                        help="Path to problem mesh file")
    parser.add_argument('-p', '--pqr', dest='pqr', type=str, default=None,
                        help="Path to problem pqr file")
    parser.add_argument('-y', '--alphay', dest='alphay', type=float, default=None,
                        help="Angle change in Y in degrees")
    parser.add_argument('-z', '--alphaz', dest='alphaz', type=float, default=None,
                        help="Angle change in Z in degrees")
    parser.add_argument('-n', '--name', dest='name', type=str, default=None,
                        help="Name to append to modified mesh file")
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


args = read_inputs()

inMesh = args.mesh
inpqr  = args.pqr
alpha_y = args.alphay*numpy.pi/180.
alpha_z = args.alphaz*numpy.pi/180.
name = args.name
prob_path = args.problem_folder
full_path = os.getcwd() + '/' + prob_path
full_path = os.path.normpath(full_path)

if name is None:
    name = ''
os.environ['PYGBE_PROBLEM_FOLDER'] = full_path

verbose = False
if args.verbose:
    verbose = True

#outMesh = inMesh+'_rot'+sys.argv[3]+'_til'+sys.argv[4]
#outpqr = inpqr+'_rot'+sys.argv[3]+'_til'+sys.argv[4]
outMesh = inMesh.split()[0] + name
outpqr = inpqr.split()[0] + name

vert = readVertex(inMesh, float)
xq, q, Nq = readpqr(full_path+'/'+inpqr, float)

#xq = numpy.array([[1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
#q = numpy.array([1.,-1.,1.])

#### Setup initial configuration
# Initial configuration: dipole parallel to y and outermost atom to center parallel to x
d = findDipole(xq,q)
normd = numpy.sqrt(numpy.sum(d*d))
normal  = numpy.array([0,1,0])
normal2 = numpy.array([1,0,0])

angle = numpy.arccos(numpy.dot(d, normal)/normd)

## Align normal and dipole vectors
# Rotate x axis
angle_x = -atan2(d[2],d[1])     # Positive angle gets away from y, then negative to take it back
xq_aux = rotate_x(xq, angle_x)
vert_aux = rotate_x(vert, angle_x)

# Rotate z axis
d_aux = findDipole(xq_aux, q)
angle_z = atan2(d_aux[0],d_aux[1]) # Positive angle approaches y, then it's ok
xq_aux2 = rotate_z(xq_aux, angle_z)
vert_aux2 = rotate_z(vert_aux, angle_z)

## Align vector of atom furthest to center to x axis
# Pick atom
ctr = numpy.average(xq_aux2, axis=0) 
r_atom = xq_aux2 - ctr
r_atom_norm = numpy.sqrt(xq_aux2[:,0]**2+xq_aux2[:,2]**2) # Distance in x-z plane
max_atom = numpy.where(r_atom_norm==max(r_atom_norm))[0][0]

# Rotate y axis
r_atom_max = r_atom[max_atom]
angle_y = atan2(r_atom_max[2], r_atom_max[0])
xq_0 = rotate_y(xq_aux2, angle_y)
vert_0 = rotate_y(vert_aux2, angle_y)

# Check if dipole and normal are parallel
d_0 = findDipole(xq_0, q)

# Check if furthest away atom vector and x axis are parallel
ctr = numpy.average(xq_0, axis=0) 
ctr[1] = xq_0[max_atom,1]
r_atom = xq_0 - ctr
max_atom_vec = r_atom[max_atom]


check_dipole = numpy.dot(d_0,numpy.array([1,1,1]))
check_atom   = numpy.dot(max_atom_vec,numpy.array([1,1,1]))
if verbose:
    print 'Initial configuration:'
if abs(check_dipole - abs(d_0[1]))<1e-10: 
    if verbose: print '\tDipole is aligned with normal'
else: print '\tDipole NOT aligned!'
if abs(check_atom - abs(max_atom_vec[0]))<1e-10: 
    if verbose: print '\tMax atom is aligned with x axis'
else: print '\tMax atom NOT aligned!'


### Move to desired configuration

## Rotate
# Rotate y axis
xq_aux = rotate_y(xq_0, alpha_y)
vert_aux = rotate_y(vert_0, alpha_y)

# Rotate z axis
xq_new = rotate_z(xq_aux, alpha_z)
vert_new = rotate_z(vert_aux, alpha_z)

## Translate
ymin = min(vert_new[:,1])
ctr = numpy.average(vert_new, axis=0) 
translation = numpy.array([ctr[0], ymin-5, ctr[2]]) # 2 Angs over the x-z plane

vert_new -= translation
xq_new -= translation

## Check
ctr = numpy.average(vert_new, axis=0) 
d = findDipole(xq_new, q)
dx = numpy.array([0, d[1], d[2]])
dy = numpy.array([d[0], 0, d[2]])
dz = numpy.array([d[0], d[1], 0])
normd = numpy.sqrt(numpy.sum(d*d))
normdx = numpy.sqrt(numpy.sum(dx*dx))
normdz = numpy.sqrt(numpy.sum(dz*dz))
angle = numpy.arccos(numpy.dot(d, normal)/normd)
anglex = numpy.arccos(numpy.dot(dx, normal)/normdx)
anglez = numpy.arccos(numpy.dot(dz, normal)/normdz)

xq_check = rotate_z(xq_new, -alpha_z)
ctr_check = numpy.average(xq_check, axis=0)
atom_vec = numpy.array([xq_check[max_atom,0]-ctr_check[0], 0, xq_check[max_atom,2]-ctr_check[2]])
atom_vec_norm = numpy.sqrt(numpy.sum(atom_vec*atom_vec))
angley = numpy.arccos(numpy.dot(atom_vec, normal2)/atom_vec_norm)

if alpha_y>numpy.pi:
    angley = 2*numpy.pi-angley    # Dot product finds the smallest angle!

if verbose:
    print 'Desired configuration:'
if abs(ctr[0])<1e-10 and abs(ctr[2])<1e-10:
    if verbose:
        print '\tProtein is centered, %f angs over the surface'%(min(vert_new[:,1]))
else:
    print '\tProtein NOT well located!'

if abs(d[2])<1e-10:
    if verbose:
        print '\tDipole is on x-y plane, %f degrees from normal'%(angle*180/numpy.pi)
else:
    print '\tDipole is NOT well aligned'

if abs(angle-alpha_z)<1e-10:
    if verbose:
        print '\tMolecule was tilted correctly by %f deg'%(angle*180/numpy.pi)
else:
    print '\tMolecule was NOT tilted correctly!'

if abs(angley-alpha_y)<1e-10:
    if verbose:
        print '\tMolecule was rotated correctly %f deg'%(angley*180/numpy.pi)
else:
    print '\tMolecule was NOT rotated correctly!'

#### Save to file
#import ipdb; ipdb.set_trace()
with open(full_path+'/'+outMesh+'.vert','w') as f:
    numpy.savetxt(f, vert_new)
cmd = 'cp '+inMesh+'.face '+outMesh+'.face'
os.system(cmd)

modifypqr(full_path+'/'+inpqr, outpqr+'.pqr', xq_new)

if verbose:
    print '\nWritten to '+outMesh+'.vert(.face) and '+outpqr+'.pqr'
