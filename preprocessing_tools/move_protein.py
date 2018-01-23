'''
Generates mesh and pqr files of a protein that has been translated and rotated.
Rotation and tilt angles are passed as arguments. 
Translation has to be modified in script.

                                    y
Axis are assumed to be oriented as   |__ x
                                    /
                                   z  
The rotation and tilt angles are passed by command line as arguments. 
The translation has to be modified in the script. 
As it is now, the protein will be located at 10 Ang of the surface of sphere of
radius 125 Ang. The displacement occurs in the z direction. In the x and y 
directions the protein is centered in 0.  
'''

import numpy
import os

from argparse import ArgumentParser

from move_prot_helper import (read_inputs, read_vertex, read_pqr, find_dipole,
                            rotate_x, rotate_y, rotate_z, modify_pqr)


#Reading the inputs:

args = read_inputs()

inMesh  = args.im
inpqr   = args.ip
alpha_y = float(args.ay)*numpy.pi/180. 
alpha_z = float(args.az)*numpy.pi/180.
name    = args.name
verbose = args.v


outMesh = inMesh + name
outpqr = inpqr + name

#Read mesh and pqr
vert = read_vertex(inMesh+'.vert', float)
xq, q, Nq, rad = read_pqr(inpqr+'.pqr', float)

#### Setup initial configuration
# Initial configuration: dipole parallel to y and outermost atom to center parallel to x
d = find_dipole(xq,q)
normd = numpy.sqrt(sum(d*d))
normal  = numpy.array([0,1,0])
normal2 = numpy.array([1,0,0])

angle = numpy.arccos(numpy.dot(d, normal)/normd)

## Align normal and dipole vectors

# Rotate x axis
angle_x = -numpy.arctan2(d[2],d[1])  # Positive angle gets away from y, then negative to take it back
xq_aux = rotate_x(xq, angle_x)
vert_aux = rotate_x(vert, angle_x)

# Rotate z axis
d_aux = find_dipole(xq_aux, q)
angle_z = numpy.arctan2(d_aux[0],d_aux[1]) # Positive angle approaches y, then it's ok
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
angle_y = numpy.arctan2(r_atom_max[2], r_atom_max[0])
xq_0 = rotate_y(xq_aux2, angle_y)
vert_0 = rotate_y(vert_aux2, angle_y)

# Check if dipole and normal are parallel
d_0 = find_dipole(xq_0, q)

# Check if furthest away atom vector and x axis are parallel
ctr = numpy.average(xq_0, axis=0) 
ctr[1] = xq_0[max_atom,1]
r_atom = xq_0 - ctr
max_atom_vec = r_atom[max_atom]


#If verbose was set to True, it'll print info about dipole alignment.

check_dipole = numpy.dot(d_0,numpy.array([1,1,1]))
check_atom   = numpy.dot(max_atom_vec,numpy.array([1,1,1]))

if verbose:
    print('Initial configuration:')
if abs(check_dipole - abs(d_0[1]))<1e-10: 
    if verbose: print ('\tDipole is aligned with normal')
else: print ('\tDipole NOT aligned!')
if abs(check_atom - abs(max_atom_vec[0]))<1e-10: 
    if verbose: print ('\tMax atom is aligned with x axis')
else: print ('\tMax atom NOT aligned!')

### Move to desired configuration

## Rotate
# Rotate y axis
xq_aux = rotate_y(xq_0, alpha_y)
vert_aux = rotate_y(vert_0, alpha_y)

# Rotate z axis
xq_new = rotate_z(xq_aux, alpha_z)
vert_new = rotate_z(vert_aux, alpha_z)

## Translate
xmin = min(vert_new[:,0])
ymin = min(vert_new[:,1])
zmin = min(vert_new[:,2])

#index of when the min in z happend 
idx_zmin = numpy.argmin(vert_new[:, 2])

R_sensor = 125 #Angstrom
dist = 10 #Angstrom
x_trans = 0
y_trans = 0
z_trans = R_sensor + dist

ctr = numpy.average(vert_new, axis=0) 

x_zmin = vert_new[idx_zmin, 0]
y_zmin = vert_new[idx_zmin, 1]


translation = numpy.array([x_zmin, y_zmin, zmin - z_trans]) # z_trans Angs apart of th sphere in the z direction

# Move according to translation

vert_new -= translation
xq_new -= translation

### Checking
r_min_last = numpy.min(numpy.linalg.norm(vert_new, axis=1))
idx_rmin_last = numpy.argmin(numpy.linalg.norm(vert_new, axis=1))


## Check
ctr = numpy.average(vert_new, axis=0) 

d = find_dipole(xq_new, q)
dx = numpy.array([0, d[1], d[2]])
dy = numpy.array([d[0], 0, d[2]])
dz = numpy.array([d[0], d[1], 0])
normd = numpy.sqrt(sum(d*d))
normdx = numpy.sqrt(sum(dx*dx))
normdz = numpy.sqrt(sum(dz*dz))
angle = numpy.arccos(numpy.dot(d, normal)/normd)
anglex = numpy.arccos(numpy.dot(dx, normal)/normdx)
anglez = numpy.arccos(numpy.dot(dz, normal)/normdz)

xq_check = rotate_z(xq_new, -alpha_z)
ctr_check = numpy.average(xq_check, axis=0)
atom_vec = numpy.array([xq_check[max_atom,0]-ctr_check[0], 0, xq_check[max_atom,2]-ctr_check[2]])
atom_vec_norm = numpy.sqrt(sum(atom_vec*atom_vec))
angley = numpy.arccos(numpy.dot(atom_vec, normal2)/atom_vec_norm)


#If verbose was set to True, it'll print isummary of configuration.

if alpha_y>numpy.pi:
    angley = 2*numpy.pi-angley    # Dot product finds the smallest angle!

if verbose:
    print ('Desired configuration:')

    print ('\tProtein is centered, {}'.format(ctr))
    print ('\tProtein r minimum is {}, located at {}'.format(r_min_last,
                                                      vert_new[idx_rmin_last, :]))

if abs(d[2])<1e-10:
    if verbose:
        print ('\tDipole is on x-y plane, %f degrees from normal'%(angle*180/numpy.pi))
else:
    print ('\tDipole is NOT well aligned')

if abs(angle-alpha_z)<1e-10:
    if verbose:
        print ('\tMolecule was tilted correctly by %f deg'%(angle*180/numpy.pi))
else:
    print ('\tMolecule was NOT tilted correctly!')

if abs(angley-alpha_y)<1e-10:
    if verbose:
        print ('\tMolecule was rotated correctly %f deg'%(angley*180/numpy.pi))
else:
    print ('\tMolecule was NOT rotated correctly!')


#### Save to file
numpy.savetxt(outMesh+'.vert', vert_new)
cmd = 'cp '+inMesh+'.face '+outMesh+'.face'
os.system(cmd)

modify_pqr(inpqr+'.pqr', outpqr+'.pqr', xq_new, q, rad)

if verbose:
    print ('\nWritten to '+outMesh+'.vert(.face) and '+outpqr+'.pqr')
