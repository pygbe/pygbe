#!/usr/bin/env python
"""
It generates a .phi0 for a sensor brick. The user has to set the values
of dphi/dn on each face of the brick. 

The surface charge \sigma is usually units of C/m^2. However, the code takes as
input dphi/dn which has to be in units of electron charge and angstrom^2.

dphi/dn = - sigma / epsilon therefore, to convert to the proper units, from sigma
to dphi/dn, we have to multiply the value of sigma by (Å)^2, divided it by the
charge of the electron q_e = 1.602x10^-19, and finally divided it by epsilon = 80 
(dielectric constant of the medium outside, usually water).

For example:

If the user desires a surface charge sigma = 0.05 C/m^2 then in this code he/she
will set up dphi/dn in the faces of the brick, where:

dphi/dn = - (0.05 x (1x10^-10)^2)/ (80 x 1.602x10^-19) = -4x10^-5
   
"""
import numpy 
import sys
import os
from pygbe.util.readData import readTriangle, readVertex
from argparse import ArgumentParser

def zeroAreas(vertex, triangle_raw, Area_null):
    """
    Looks for "zero-areas", areas that are really small, almost zero. It appends
    them to Area_null list.
    """
    for i in range(len(triangle_raw)):
        L0 = vertex[triangle_raw[i,1]] - vertex[triangle_raw[i,0]]
        L2 = vertex[triangle_raw[i,0]] - vertex[triangle_raw[i,2]]
        normal_aux = numpy.cross(L0,L2)
        Area_aux = numpy.linalg.norm(normal_aux)/2
        if Area_aux<1e-10:
            Area_null.append(i)
    return Area_null 

## Designed for a cube which faces are aligned with cartesian coordinates

def read_inputs():
    """
    Parse command-line arguments to generate_phi0.
    User should provide:
    - Problem folder (can be inferred from files if not provided)
    - Mesh file (without .vert or .face) which phi0 is desired.  
    - x_right : value of dphi/dn in the x_right face.
    - x_left  : value of dphi/dn in the x_left face.
    - y_top   : value of dphi/dn in the y_top face.
    - y_bottom: value of dphi/dn in the y_bottom face.  
    - z_front : value of dphi/dn in the z_front face.
    - z_back  : value of dphi/dn in the z_back face.
    """

    parser = ArgumentParser(description='Manage generate_phi0 command line arguments')

    parser.add_argument('problem_folder', type=str,
                        help="Path to folder containing problem files")

    parser.add_argument('-m', '--mesh', dest='mesh', type=str, default=None,
                        help="Path to sensor-brick mesh file")

    parser.add_argument('-x_r', '--x_right', dest='x_right', type=float, default=None,
                        help="charge assigned to x_right face")

    parser.add_argument('-x_l', '--x_left', dest='x_left', type=float, default=None,
                        help="charge assigned to x_left face")

    parser.add_argument('-y_t', '--y_top', dest='y_top', type=float, default=None,
                        help="charge assigned to y_top face")

    parser.add_argument('-y_b', '--y_bottom', dest='y_bottom', type=float, default=None,
                        help="charge assigned to y_bottom face")

    parser.add_argument('-z_f', '--z_front', dest='z_front', type=float, default=None,
                        help="charge assigned to z_front face")

    parser.add_argument('-z_b', '--z_back', dest='z_back', type=float, default=None,
                        help="charge assigned to z_back face")

    return parser.parse_args()


args = read_inputs()

meshFile = args.mesh

x_right = args.x_right
x_left  = args.x_left
y_top   = args.y_top
y_bott  = args.y_bottom
z_front = args.z_front
z_back  = args.z_back

full_path = args.problem_folder

if not os.path.isdir(full_path):
    full_path = os.getcwd() + '/' + full_path
full_path = os.path.normpath(full_path)

os.environ['PYGBE_PROBLEM_FOLDER'] = full_path


vertex = readVertex(meshFile+'.vert', float)
triangle_raw = readTriangle(meshFile+'.face', 'neumann_surface')

Area_null = []
Area_null = zeroAreas(vertex, triangle_raw, Area_null)
triangle = numpy.delete(triangle_raw, Area_null, 0)

if len(triangle) != len(triangle_raw):
    print '%i deleted triangles'%(len(triangle_raw)-len(triangle))

phi0 = numpy.zeros(len(triangle), float)

tri_ctr = numpy.average(vertex[triangle], axis=1)
print len(tri_ctr)
print len(triangle)

max_x = max(tri_ctr[:,0])
min_x = min(tri_ctr[:,0])
max_y = max(tri_ctr[:,1])
min_y = min(tri_ctr[:,1])
max_z = max(tri_ctr[:,2])
min_z = min(tri_ctr[:,2])

for i in range(len(triangle)):
    if abs(tri_ctr[i,0]-max_x)<1e-10:
        phi0[i] = x_right
    if abs(tri_ctr[i,0]-min_x)<1e-10:
        phi0[i] = x_left 
    if abs(tri_ctr[i,1]-max_y)<1e-10:
        phi0[i] = y_top  
    if abs(tri_ctr[i,1]-min_y)<1e-10:
        phi0[i] = y_bott 
    if abs(tri_ctr[i,2]-max_z)<1e-10:
        phi0[i] = z_front
    if abs(tri_ctr[i,2]-min_z)<1e-10:
        phi0[i] = z_back 


meshFile = meshFile.rsplit('/', 1)[-1]

faces_values = [x_right, x_left, y_top, y_bott, z_front, z_back]
faces_values = ''.join([str(face) for face in faces_values])

file_out = meshFile+'_'+faces_values+'.phi0'

with open(full_path+'/'+file_out, 'w') as f:
    numpy.savetxt(f, phi0)

