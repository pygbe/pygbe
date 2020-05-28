'''
Rotates the protein by a solid angle on the plane xz
'''

import numpy
import os

from argparse import ArgumentParser

from move_prot_helper import (read_vertex, read_pqr, rotate_y, rotate_x,
                             modify_pqr)

def read_inputs():
    """
    Parse command-line arguments to run move_protein.

    User should provide:
    -inMesh : str, mesh file you want to rotate.
    -inpqr  : str, pqr of the object you want to rotate.
    -alpha_y: float [degrees], rotation angle, about the dipole moment. 
    -name   : str, output file name.
    """

    parser = ArgumentParser(description='Manage solid_rotation_y command line arguments')


    parser.add_argument('-im', '--inMesh', dest='im', type=str, default=None,
                        help="mesh file you want to rotate")

    parser.add_argument('-ip', '--inpqr', dest='ip', type=str, default=None,
                        help="pqr of the object you want to rotate")

    parser.add_argument('-angy', '--angle_y', dest='angy', type=float, default=None,
                        help="rotation angle in the plane xz")

    parser.add_argument('-n', '--name', dest='name', type=str, default='',
                        help="output file name")
    
    return parser.parse_args()

args = read_inputs()

inMesh  = args.im
inpqr   = args.ip
angle_y = float(args.angy)*numpy.pi/180. 
name    = args.name

outMesh = inMesh + name
outpqr = inpqr + name

#Read mesh and pqr
vert = numpy.loadtxt(inMesh+'.vert', dtype=float)

xq, q, Nq, rad = read_pqr(inpqr+'.pqr', float)

#Comment if want to rotate respect to x axis. 
xq_new = rotate_y(xq, angle_y)
vert_new = rotate_y(vert, angle_y)

# If desired to rotate on plane yz use function rotate_x, comment 2 lines above
#Uncomment the next two lines if rotation is desired around x-axis
#xq_new = rotate_x(xq, angle_y)
#vert_new = rotate_x(vert, angle_y)

ctr = numpy.average(vert_new, axis=0) 

r_min_last = numpy.min(numpy.linalg.norm(vert_new, axis=1))
idx_rmin_last = numpy.argmin(numpy.linalg.norm(vert_new, axis=1))

print ('Desired configuration:')

print ('\tProtein is centered, {}'.format(ctr))
print ('\tProtein r minimum is {}, located at {}'.format(r_min_last,
                                                      vert_new[idx_rmin_last, :]))

#### Save to file
numpy.savetxt(outMesh+'.vert', vert_new)
cmd = 'cp '+inMesh+'.face '+outMesh+'.face'
os.system(cmd)

modify_pqr(inpqr+'.pqr', outpqr+'.pqr', xq_new, q, rad)

print ('\nWritten to '+outMesh+'.vert(.face) and '+outpqr+'.pqr')
