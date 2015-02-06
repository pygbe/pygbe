#!/usr/bin/env python
# Calculated according to FelderPriluskySilmanSussman2007, but using center of mass
from numpy import *
from math import atan2

import os
import sys
sys.path.append('../util')
from readData import readVertex, readpqr

def findDipole(xq, q):

    ctr = sum(transpose(xq)*abs(q), axis=1)/sum(abs(q))
#    ctr = average(xq, axis=0)
    r = xq - ctr
    d = sum(transpose(r)*q, axis=1)

    return d

def rotate_x(x, angle):

    xnew = zeros(shape(x))
    xnew[:,0] = x[:,0]
    xnew[:,1] = x[:,1]*cos(angle) - x[:,2]*sin(angle)
    xnew[:,2] = x[:,1]*sin(angle) + x[:,2]*cos(angle)

    return xnew


def rotate_y(x, angle):

    xnew = zeros(shape(x))
    xnew[:,0] = x[:,2]*sin(angle) + x[:,0]*cos(angle)
    xnew[:,1] = x[:,1]
    xnew[:,2] = x[:,2]*cos(angle) - x[:,0]*sin(angle)

    return xnew    

def rotate_z(x, angle):

    xnew = zeros(shape(x))
    xnew[:,0] = x[:,0]*cos(angle) - x[:,1]*sin(angle)
    xnew[:,1] = x[:,0]*sin(angle) + x[:,1]*cos(angle)
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




inMesh = sys.argv[1]
inpqr  = sys.argv[2]
alpha_y = float(sys.argv[3])*pi/180.
alpha_z = float(sys.argv[4])*pi/180.
if len(sys.argv)>5:
    name = sys.argv[5]
else:
    name = ''
if len(sys.argv)>6:
    if sys.argv[6] == 'verbose':
        verbose = True
else:
    verbose = False

#outMesh = inMesh+'_rot'+sys.argv[3]+'_til'+sys.argv[4]
#outpqr = inpqr+'_rot'+sys.argv[3]+'_til'+sys.argv[4]
outMesh = inMesh + name
outpqr = inpqr + name

vert = readVertex(inMesh+'.vert', float)
xq, q, Nq = readpqr(inpqr+'.pqr', float)

#xq = array([[1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
#q = array([1.,-1.,1.])

#### Setup initial configuration
# Initial configuration: dipole parallel to y and outermost atom to center parallel to x
d = findDipole(xq,q)
normd = sqrt(sum(d*d))
normal  = array([0,1,0])
normal2 = array([1,0,0])

angle = arccos(dot(d, normal)/normd)

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
ctr = average(xq_aux2, axis=0) 
r_atom = xq_aux2 - ctr
r_atom_norm = sqrt(xq_aux2[:,0]**2+xq_aux2[:,2]**2) # Distance in x-z plane
max_atom = where(r_atom_norm==max(r_atom_norm))[0][0]

# Rotate y axis
r_atom_max = r_atom[max_atom]
angle_y = atan2(r_atom_max[2], r_atom_max[0])
xq_0 = rotate_y(xq_aux2, angle_y)
vert_0 = rotate_y(vert_aux2, angle_y)

# Check if dipole and normal are parallel
d_0 = findDipole(xq_0, q)

# Check if furthest away atom vector and x axis are parallel
ctr = average(xq_0, axis=0) 
ctr[1] = xq_0[max_atom,1]
r_atom = xq_0 - ctr
max_atom_vec = r_atom[max_atom]


check_dipole = dot(d_0,array([1,1,1]))
check_atom   = dot(max_atom_vec,array([1,1,1]))
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
ctr = average(vert_new, axis=0) 
translation = array([ctr[0], ymin-5, ctr[2]]) # 2 Angs over the x-z plane

vert_new -= translation
xq_new -= translation

## Check
ctr = average(vert_new, axis=0) 
d = findDipole(xq_new, q)
dx = array([0, d[1], d[2]])
dy = array([d[0], 0, d[2]])
dz = array([d[0], d[1], 0])
normd = sqrt(sum(d*d))
normdx = sqrt(sum(dx*dx))
normdz = sqrt(sum(dz*dz))
angle = arccos(dot(d, normal)/normd)
anglex = arccos(dot(dx, normal)/normdx)
anglez = arccos(dot(dz, normal)/normdz)

xq_check = rotate_z(xq_new, -alpha_z)
ctr_check = average(xq_check, axis=0)
atom_vec = array([xq_check[max_atom,0]-ctr_check[0], 0, xq_check[max_atom,2]-ctr_check[2]])
atom_vec_norm = sqrt(sum(atom_vec*atom_vec))
angley = arccos(dot(atom_vec, normal2)/atom_vec_norm)

if alpha_y>pi:
    angley = 2*pi-angley    # Dot product finds the smallest angle!

if verbose:
    print 'Desired configuration:'
if abs(ctr[0])<1e-10 and abs(ctr[2])<1e-10:
    if verbose:
        print '\tProtein is centered, %f angs over the surface'%(min(vert_new[:,1]))
else:
    print '\tProtein NOT well located!'

if abs(d[2])<1e-10:
    if verbose:
        print '\tDipole is on x-y plane, %f degrees from normal'%(angle*180/pi)
else:
    print '\tDipole is NOT well aligned'

if abs(angle-alpha_z)<1e-10:
    if verbose:
        print '\tMolecule was tilted correctly by %f deg'%(angle*180/pi)
else:
    print '\tMolecule was NOT tilted correctly!'

if abs(angley-alpha_y)<1e-10:
    if verbose:
        print '\tMolecule was rotated correctly %f deg'%(angley*180/pi)
else:
    print '\tMolecule was NOT rotated correctly!'

#### Save to file
savetxt(outMesh+'.vert', vert_new)
cmd = 'cp '+inMesh+'.face '+outMesh+'.face'
os.system(cmd)

modifypqr(inpqr+'.pqr', outpqr+'.pqr', xq_new)

if verbose:
    print '\nWritten to '+outMesh+'.vert(.face) and '+outpqr+'.pqr'
