"""
It contains the functions to read the all the data from the meshes files, its
parameters and charges files. 
"""

import numpy
import os



def readVertex(filename, REAL):
    """
    It reads the vertex of the triangles from the mesh file and it stores
    them on an array.

    Arguments:
    ----------
    filename: name of the file that contains the surface information.
              (filename.vert)
    REAL    : data type.
    
    Returns:
    -------
    vertex: array, vertices of the triangles.
    """
    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')
    geo_path = os.environ.get('PYGBE_GEOMETRY')
    if geo_path:
        full_path = geo_path
    X = numpy.loadtxt(os.path.join(full_path, filename), dtype=REAL)
    vertex = X[:, 0:3]

    return vertex


def readVertex2(filename, REAL):
    """
    It reads the vertex of the triangles from the mesh file and it stores
    them on an array. 
    It reads the file line by line.

    Arguments:
    ----------
    filename: name of the file that contains the surface information.
              (filename.vert)
    REAL    : data type.
    
    Returns:
    -------
    vertex: array, vertices of the triangles.
    """

    x = []
    y = []
    z = []
    for line in file(filename):
        line = line.split()
        x0 = line[0]
        y0 = line[1]
        z0 = line[2]
        x.append(REAL(x0))
        y.append(REAL(y0))
        z.append(REAL(z0))

    x = numpy.array(x)
    y = numpy.array(y)
    z = numpy.array(z)
    vertex = numpy.zeros((len(x), 3))
    vertex[:, 0] = x
    vertex[:, 1] = y
    vertex[:, 2] = z

    return vertex


def readTriangle(filename, surf_type):
    """
    It reads the triangles from the mesh file and it stores them on an
    array.

    Arguments:
    ----------
    filename : name of the file that contains the surface information.
               (filename.faces)
    surf_type: str, type of surface.
    
    Returns:
    -------
    triangle: array, triangles indices.
    """

    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')
    geo_path = os.environ.get('PYGBE_GEOMETRY')
    if geo_path:
        full_path = geo_path
    X = numpy.loadtxt(os.path.join(full_path, filename), dtype=int)
    triangle = numpy.zeros((len(X), 3), dtype=int)
    #    if surf_type<=10:
    if surf_type == 'internal_cavity':
        triangle[:, 0] = X[:, 0]
        triangle[:, 1] = X[:, 1]
        triangle[:, 2] = X[:, 2]
    else:
        triangle[:, 0] = X[:, 0]
        triangle[:,
                 1] = X[:, 2
                        ]  # v2 and v3 are flipped to match my sign convention!
        triangle[:, 2] = X[:, 1]

    triangle -= 1

    return triangle


def readTriangle2(filename):
    """
    It reads the triangles from the mesh file and it stores them on an
    array.
    It reads the file line by line.

    Arguments:
    ----------
    filename : name of the file that contains the surface information.
               (filename.faces)
    
    Returns:
    -------
    triangle: array, triangles indices.
    """

    triangle = []

    for line in file(filename):
        line = line.split()
        v1 = line[0]
        v2 = line[2]  # v2 and v3 are flipped to match my sign convention!
        v3 = line[1]
        triangle.append([int(v1) - 1, int(v2) - 1, int(v3) - 1])
        # -1-> python starts from 0, matlab from 1

    triangle = numpy.array(triangle)

    return triangle



def readCheck(aux, REAL):
    """
    It checks if it is not reading more than one term.
    We use this function to check we are not missing '-' signs in the .pqr
    files, when we read the lines.

    Arguments:
    ----------
    aux : str, string to be checked.
    REAL: data type.
    
    
    Returns:
    -------
    X: array, array with the correct '-' signs assigned.
    """

    cut = [0]
    i = 0
    for c in aux[1:]:
        i += 1
        if c == '-':
            cut.append(i)
    cut.append(len(aux))
    X = numpy.zeros(len(cut) - 1)
    for i in range(len(cut) - 1):
        X[i] = REAL(aux[cut[i]:cut[i + 1]])

    return X


def readpqr(filename, REAL):
    """
    It reads the pqr file, file that contains the charges information.

    Arguments:
    ----------
    filename: name of the file that contains the surface information.
               (filename.pqr)
    REAL    : data type.
    
    Returns:
    -------
    pos     : (Nqx3) array, positions of the charges. 
    q       : (Nqx1) array, value of the charges. 
    Nq      : int, number of charges.
    """

    pos = []
    q = []
    for line in file(filename):
        line = numpy.array(line.split())
        line_aux = []

        if line[0] == 'ATOM':
            for l in range(len(line) - 6):
                aux = line[5 + len(line_aux)]
                if len(aux) > 14:
                    X = readCheck(aux, REAL)
                    for i in range(len(X)):
                        line_aux.append(X[i])
#                        line_test.append(str(X[i]))
                else:
                    #                    line_test.append(line[5+len(line_aux)])
                    line_aux.append(REAL(line[5 + len(line_aux)]))

#            line_test.append(line[len(line)-1])
            x = line_aux[0]
            y = line_aux[1]
            z = line_aux[2]
            q.append(line_aux[3])
            pos.append([x, y, z])

#           for i in range(10):
#                f.write("%s\t"%line_test[i])
#            f.write("\n")

#    f.close()
#    quit()
    pos = numpy.array(pos)
    q = numpy.array(q)
    Nq = len(q)
    return pos, q, Nq


def readcrd(filename, REAL):
    """
    It reads the crd file, file that contains the charges information.

    Arguments:
    ----------
    filename : name of the file that contains the surface information.
    REAL: data type.
    
    Returns:
    -------
    pos: (Nqx3) array, positions of the charges. 
    q  : (Nqx1) array, value of the charges. 
    Nq : int, number of charges.
    """

    pos = []
    q = []

    start = 0
    for line in file(filename):
        line = numpy.array(line.split())

        if len(line) > 8 and line[0] != '*':  # and start==2:
            x = line[4]
            y = line[5]
            z = line[6]
            q.append(REAL(line[9]))
            pos.append([REAL(x), REAL(y), REAL(z)])
        '''
        if len(line)==1:
            start += 1
            if start==2:
                Nq = int(line[0])
        '''
    pos = numpy.array(pos)
    q = numpy.array(q)
    Nq = len(q)
    return pos, q, Nq


def readParameters(param, filename):
    """
    It reads the parameters    .

    Arguments:
    ----------
    param   : class, parameters related to the surface.     
    filename: name of the file that contains the parameters information. 
              (filename.param)
        
    Returns:
    -------
    dataType:
    """

    val = []
    for line in file(filename):
        line = line.split()
        val.append(line[1])

    dataType = val[0]  # Data type
    if dataType == 'double':
        param.REAL = numpy.float64
    elif dataType == 'float':
        param.REAL = numpy.float32

    REAL = param.REAL
    param.K = int(val[1])  # Gauss points per element
    param.Nk = int(val[2])  # Number of Gauss points per side
    # for semi analytical integral
    param.K_fine = int(val[3])  # Number of Gauss points per element
    # for near singular integrals
    param.threshold = REAL(val[4])  # L/d threshold to use analytical integrals
    # Over: analytical, under: quadrature
    param.BSZ = int(val[5])  # CUDA block size
    param.restart = int(val[6])  # Restart for GMRES
    param.tol = REAL(val[7])  # Tolerance for GMRES
    param.max_iter = int(val[8])  # Max number of iteration for GMRES
    param.P = int(val[9])  # Order of Taylor expansion for treecode
    param.eps = REAL(val[10])  # Epsilon machine
    param.NCRIT = int(val[11])  # Max number of targets per twig box of tree
    param.theta = REAL(val[12])  # MAC criterion for treecode
    param.GPU = int(val[13])  # =1: use GPU, =0 no GPU

    return dataType


def readFields(filename):  
    """
    It reads the physical parameters from the configuration file for each region
    in the surface and it appends them on the corresponding list.

    Arguments:
    ----------
    filename: name of the file that contains the physical parameters of each
              region. (filename.config)
        
    Returns:
    -------
    LorY    : list, it contains integers, Laplace (1) or Yukawa (2),
                    corresponding to each region.
    pot     : list, it contains integers indicating to calculate (1) or not (2)
                    the energy in this region.
    E       : list, it contains floats with the dielectric constant of each
                    region.
    kappa   : list, it contains floats with the reciprocal of Debye length value
                    of each region.
    charges : list, it contains integers indicating if there are (1) or not (0)
                    charges in this region.
    coulomb : list, it contains integers indicating to calculate (1) or not (2)
                    the coulomb energy in this region.
    qfile   : list, location of the '.pqr' file with the location of the charges.    
    Nparent : list, it contains integers indicating the number of 'parent'
                    surfaces (surface containing this region)
    parent  : list, it contains the file of the parent surface mesh, according 
                    to their position in the FILE list, starting from 0 (eg. if
                    the mesh file for the parent is the third one specified in 
                    the FILE section, parent=2)
    Nchild  : list, it contains integers indicating number of child surfaces
                    (surfaces completely contained in this region).
    child   :  list, it contains position of the mesh files for the children 
                     surface in the FILE section.
    """

    LorY = []
    pot = []
    E = []
    kappa = []
    charges = []
    coulomb = []
    qfile = []
    Nparent = []
    parent = []
    Nchild = []
    child = []

    for line in file(filename):
        line = line.split()
        if len(line) > 0:
            if line[0] == 'FIELD':
                LorY.append(line[1])
                pot.append(line[2])
                E.append(line[3])
                kappa.append(line[4])
                charges.append(line[5])
                coulomb.append(line[6])
                qfile.append(line[7] if line[7] == 'NA' else
                    os.path.join(os.environ.get('PYGBE_PROBLEM_FOLDER'), line[7]))
                Nparent.append(line[8])
                parent.append(line[9])
                Nchild.append(line[10])
                for i in range(int(Nchild[-1])):
                    child.append(line[11 + i])

    return LorY, pot, E, kappa, charges, coulomb, qfile, Nparent, parent, Nchild, child


def read_surface(filename):
    """
    It reads the type of surface of each region on the surface from the 
    configuration file.

    Arguments:
    ----------
    filename: name of the file that contains the surface type of each region.
              (filename.config)  
    Returns:
    -------
    files    : list, it contains the files corresponding to each region in the
                     surface.
    surf_type: list, it contains the type of surface of each region.
    phi0_file: list, it contains the constant potential/surface charge for the
                     cases where it applies. (dirichlet or neumann surfaces)
    """

    files = []
    surf_type = []
    phi0_file = []
    for line in file(filename):
        line = line.split()
        if len(line) > 0:
            if line[0] == 'FILE':
                files.append(line[1])
                surf_type.append(line[2])
                if (line[2] == 'dirichlet_surface' or
                        line[2] == 'neumann_surface' or
                        line[2] == 'neumann_surface_hyper'):
                    phi0_file.append(os.path.join(
                        os.environ.get('PYGBE_PROBLEM_FOLDER'), line[3]))
                else:
                    phi0_file.append('no_file')

    return files, surf_type, phi0_file
