#!/usr/bin/env python
"""
Creates a sphere of radius r, centered at x0, y0, z0

Arguments (command line arguments)
----------
rec : int, number of recursions for unit sphere.
r   : float, radius.
x0  : float, x center of sphere.
y0  : float, y center of sphere.
z0  : float, z center of sphere.
name: str, output file name.
 
Returns
-----
File with vertices ".vert"
File with triangle indices ".face"
"""

import numpy
from triangulation import create_unit_sphere
from argparse import ArgumentParser

def read_inputs():
    """
    Parse command-line arguments to run mesh_sphere.

    User should provide:
    -rec : int, number of recursions for unit sphere.
    -r   : float, radius of the sphere.
    -x0  : float, x coordinate of the center of sphere.
    -y0  : float, y coordinate of the center of sphere.
    -z0  : float, z coordinate of the center of sphere.
    -name: str, output file name.    
    """

    parser = ArgumentParser(description='Manage mesh_sphere command line arguments')


    parser.add_argument('-rec', '--recursions', dest='rec', type=int, default=None,
                        help="number of recursions for unit sphere")

    parser.add_argument('-r', '--radius', dest='r', type=float, default=None,
                        help="radius of the sphere")

    parser.add_argument('-x0', '--x_center', dest='x0', type=float, default=None,
                        help="x coordinate of the center of sphere")

    parser.add_argument('-y0', '--y_center', dest='y0', type=float, default=None,
                        help="y coordinate of the center of sphere")

    parser.add_argument('-z0', '--z_center', dest='z0', type=float, default=None,
                        help="z coordinate of the center of sphere")

    parser.add_argument('-n', '--name', dest='name', type=str, default=None,
                        help="output file name")

    return parser.parse_args()

args = read_inputs()

rec      = args.rec
r        = args.r
x0       = args.x0
y0       = args.y0
z0       = args.z0
filename = args.name

xc = numpy.array([x0,y0,z0])
vertex, index, center = create_unit_sphere(rec)
vertex *= r
vertex += xc

index += 1 # Agrees with msms format
index_format = numpy.zeros_like(index)
index_format[:,0] = index[:,0]
index_format[:,1] = index[:,2]
index_format[:,2] = index[:,1]

# Check
x_test = numpy.average(vertex[:,0])
y_test = numpy.average(vertex[:,1])
z_test = numpy.average(vertex[:,2])
if abs(x_test-x0)>1e-12 or abs(y_test-y0)>1e-12 or abs(z_test-z0)>1e-12:
    print 'Center is not right!'

numpy.savetxt(filename+'.vert', vertex, fmt='%.4f')
numpy.savetxt(filename+'.face', index_format, fmt='%i')

print 'Sphere with %i faces, radius %f and centered at %f,%f,%f was saved to the file '%(len(index), r, x0, y0, z0)+filename
