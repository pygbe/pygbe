## This helper contains a set of functions used in move_protein script

import numpy

from argparse import ArgumentParser


def read_inputs():
    """
    Parse command-line arguments to run move_protein.

    User should provide:
    -inMesh : str, mesh file you want to rotate.
    -inpqr  : str, pqr of the object you want to rotate.
    -alpha_y: float [degrees], rotation angle, about the dipole moment. 
    -alpha_z: float [degrees], tilt angle.
    -name   : str, output file name.
    -verbose: bool, set to True if extra information is desired.
    """

    parser = ArgumentParser(description='Manage move_protein command line arguments')


    parser.add_argument('-im', '--inMesh', dest='im', type=str, default=None,
                        help="mesh file you want to rotate")

    parser.add_argument('-ip', '--inpqr', dest='ip', type=str, default=None,
                        help="pqr of the object you want to rotate")

    parser.add_argument('-ay', '--alpha_y', dest='ay', type=float, default=None,
                        help="rotation angle, about the dipole moment")

    parser.add_argument('-az', '--alpha_z', dest='az', type=float, default=None,
                        help="tilt angle")

    parser.add_argument('-n', '--name', dest='name', type=str, default='',
                        help="output file name")
    
    parser.add_argument('-v', '--verbose', dest='v', type=bool, default=False,
                        help="set to True if extra information is desired")

    return parser.parse_args()



def read_vertex(filename, REAL):
    """
    It reads the vertex of the triangles from the mesh file and it stores
    them on an array.

    Arguments
    ----------
    filename: name of the file that contains the surface information.
              (filename.vert)
    REAL    : data type.

    Returns
    -------
    vertex: array, vertices of the triangles.
    """

    X = numpy.loadtxt(filename, dtype=REAL)
    vertex = X[:, 0:3]

    return vertex

def read_pqr(filename, REAL):
    """
    Read charge information from pqr file

    Arguments
    ----------
    filename: name of the file that contains the surface information.
               (filename.pqr)
    REAL    : data type.

    Returns
    -------
    pos     : (Nqx3) array, positions of the charges.
    q       : (Nqx1) array, value of the charges.
    Nq      : float, number of charges, length of array q. 
    rad      : (Nqx1) array, value of the radius of the charges.   
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    pos = []
    q = []
    rad = []
    for line in lines:
        line = line.split()

        if line[0] == 'ATOM':
            #  grab coordinates and charge from columns
            x, y, z, q0, r0 = [REAL(i) for i in line[5:]]
            q.append(q0)
            rad.append(r0)
            pos.append([x, y, z])

    pos = numpy.array(pos)
    q = numpy.array(q)
    rad = numpy.array(rad)
    Nq = len(q)

    return pos, q, Nq, rad 


def find_dipole(xq, q):
    """
    Finds the dipole moment of the protein. 

    Arguments
    ---------
    xq      : (Nqx3) array, positions of the charges.
    q       : (Nqx1) array, value of the charges.

    Returns
    -------
    d       : array, xyz cordinates of the dipole moment. 

    """

    ctr = numpy.sum(numpy.transpose(xq)*abs(q), axis=1)/numpy.sum(abs(q))
    r = xq - ctr

    d = numpy.sum(numpy.transpose(r)*q, axis=1)

    return d


def rotate_x(pos, angle):
    """ Rotates the coordinates respect to x-axis, by an angle. 
    
    Arguments
    ---------
    pos   : (Nx3) array, positions.
    angle : float, angle in radians.  

    Return
    ------
    posnew  : (Nx3) array, rotated positions.

    """

    posnew = numpy.zeros(numpy.shape(pos))
    posnew[:,0] = pos[:,0]
    posnew[:,1] = pos[:,1]*numpy.cos(angle) - pos[:,2]*numpy.sin(angle)
    posnew[:,2] = pos[:,1]*numpy.sin(angle) + pos[:,2]*numpy.cos(angle)

    return posnew

def rotate_y(pos, angle):
    """ Rotates the coordinates respect to y-axis, by an angle. 
    
    Arguments
    ---------
    pos   : (Nx3) array, positions.
    angle : float, angle in radians.  

    Return
    ------
    posnew  : (Nx3) array, rotated positions.

    """

    posnew = numpy.zeros(numpy.shape(pos))
    posnew[:,0] = pos[:,2]*numpy.sin(angle) + pos[:,0]*numpy.cos(angle)
    posnew[:,1] = pos[:,1]
    posnew[:,2] = pos[:,2]*numpy.cos(angle) - pos[:,0]*numpy.sin(angle)

    return posnew   

def rotate_z(pos, angle):
    """ Rotates the coordinates respect to z-axis, by an angle. 
    
    Arguments
    ---------
    pos   : (Nx3) array, positions.
    angle : float, angle in radians.  

    Return
    ------
    posnew  : (Nx3) array, rotated positions.

    """

    posnew = numpy.zeros(numpy.shape(pos))
    posnew[:,0] = pos[:,0]*numpy.cos(angle) - pos[:,1]*numpy.sin(angle)
    posnew[:,1] = pos[:,0]*numpy.sin(angle) + pos[:,1]*numpy.cos(angle)
    posnew[:,2] = pos[:,2]

    return posnew   

def modify_pqr(inpqr, outpqr, xq, q, rad):
    
    with open(outpqr, 'w') as file_o:
        atm_nu = -1
        with open(inpqr, 'r') as file_i:
            for line in file_i:
                line_split = line.split()
                if line_split[0] == 'ATOM':
                    atm_nu += 1
                    separator = '  '
                    line_beg = separator.join(line_split[:5])
                    
                    line_add = '  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f} \n'.format(xq[atm_nu,0], 
                                                                xq[atm_nu,1],
                                                                xq[atm_nu,2], 
                                                                q[atm_nu], 
                                                                rad[atm_nu])
                    
                    line_new = line_beg + line_add 
                    file_o.write(line_new)
                    atm_nu
                else:
                    file_o.write(line)
