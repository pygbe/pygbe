"""
It contains the functions to read the all the data from the meshes files, its
parameters and charges files.
"""
import os
import numpy


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

    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')
    geo_path = os.environ.get('PYGBE_GEOMETRY')
    if geo_path:
        full_path = geo_path
    X = numpy.loadtxt(os.path.join(full_path, filename), dtype=REAL)
    vertex = X[:, 0:3]

    return vertex


def read_triangle(filename, surf_type):
    """
    It reads the triangles from the mesh file and it stores them on an
    array.

    Arguments
    ----------
    filename : name of the file that contains the surface information.
               (filename.faces)
    surf_type: str, type of surface.

    Returns
    -------
    triangle: array, triangles indices.
    """

    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')
    geo_path = os.environ.get('PYGBE_GEOMETRY')
    if geo_path:
        full_path = geo_path
    X = numpy.loadtxt(os.path.join(full_path, filename), dtype=int)
    if surf_type == 'internal_cavity':
        triangle = X[:, :3]
    else:
        # v2 and v3 are flipped to match my sign convention!
        triangle = X[:, (0, 2, 1)]

    triangle -= 1

    return triangle


def readCheck(aux, REAL):
    """
    It checks if it is not reading more than one term.
    We use this function to check we are not missing '-' signs in the .pqr
    files, when we read the lines.

    Arguments
    ----------
    aux : str, string to be checked.
    REAL: data type.


    Returns
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
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    pos = []
    q = []
    for line in lines:
        line = line.split()

        if line[0] == 'ATOM':
            #  grab coordinates and charge from columns
            x, y, z, q0 = [REAL(i) for i in line[5:-1]]
            q.append(q0)
            pos.append([x, y, z])

    pos = numpy.array(pos)
    q = numpy.array(q)
    return pos, q


def readcrd(filename, REAL):
    """
    It reads the crd file, file that contains the charges information.

    Arguments
    ----------
    filename : name of the file that contains the surface information.
    REAL     : data type.

    Returns
    -------
    pos      : (Nqx3) array, positions of the charges.
    q        : (Nqx1) array, value of the charges.
    Nq       : int, number of charges.
    """

    pos = []
    q = []

    start = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()

            if len(line) > 8 and line[0] != '*':  # and start==2:
                x = line[4]
                y = line[5]
                z = line[6]
                q.append(REAL(line[9]))
                pos.append([REAL(x), REAL(y), REAL(z)])

    pos = numpy.array(pos)
    q = numpy.array(q)
    return pos, q


def read_parameters(param, filename):
    """
    It populates the attributes from the Parameters class with the information
    read from the .param file.

    Arguments
    ----------
    param   : class, parameters related to the surface.
    filename: name of the file that contains the parameters information.
              (filename.param)

    Returns
    -------
    dataType: we return the dataType of each attributes because we need it for
              other fucntions.
    """

    val = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
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


def read_fields(filename):
    """
    Read the physical parameters from the configuration file for each region
    in the surface

    Arguments
    ----------
    filename: name of the file that contains the physical parameters of each
              region. (filename.config)

    Returns
    -------
    Dictionary containing:
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

    field = dict()
    field['LorY'] = []
    field['pot'] = []
    field['E'] = []
    field['kappa'] = []
    field['charges'] = []
    field['coulomb'] = []
    field['qfile'] = []
    field['Nparent'] = []
    field['parent'] = []
    field['Nchild'] = []
    field['child'] = []

    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        if len(line) > 0:
            if line[0] == 'FIELD':
                field['LorY'].append(line[1])
                field['pot'].append(line[2])
                field['E'].append(line[3])
                field['kappa'].append(line[4])
                field['charges'].append(line[5])
                field['coulomb'].append(line[6])
                field['qfile'].append(line[7] if line[7] == 'NA' else
                    os.path.join(os.environ.get('PYGBE_PROBLEM_FOLDER'), line[7]))
                field['Nparent'].append(line[8])
                field['parent'].append(line[9])
                field['Nchild'].append(line[10])
                for i in range(int(field['Nchild'][-1])):
                    field['child'].append(line[11 + i])

    for key in ['LorY', 'Nparent', 'parent', 'Nchild', 'child',
                'pot', 'coulomb', 'charges']:
        field[key] = [int(i) if i != 'NA' else 0 for i in field[key]]

    # NOTE: We ignore the value of `param.REAL` here but cast down in
    # `class_initialization.initialize_field` if necessary
    field['E'] = [complex(i) if 'j' in i else float(i) if i != 'NA' else 0
        for i in field['E']]
    field['kappa'] = [float(i) if i != 'NA' else 0 for i in field['kappa']]

    return field


def read_surface(filename):
    """
    It reads the type of surface of each region on the surface from the
    configuration file.

    Arguments
    ---------
    filename: name of the file that contains the surface type of each region.
              (filename.config).

    Returns
    -------
    files    : list, it contains the files corresponding to each region in the
                     surface.
    surf_type: list, it contains the type of surface of each region.
    phi0_file: list, it contains the constant potential/surface charge for the
                     cases where it applies. (dirichlet or neumann surfaces)
    """
    surfaces_req_phi0= ['dirichlet_surface',
                        'neumann_surface',
                        'neumann_surface_hyper']
    files = []
    surf_type = []
    phi0_file = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        if line:
            if line[0] == 'FILE':
                files.append(line[1])
                surf_type.append(line[2])
                if line[2] in surfaces_req_phi0:
                    phi0_file.append(os.path.join(
                        os.environ.get('PYGBE_PROBLEM_FOLDER'), line[3]))
                else:
                    phi0_file.append('no_file')

    return files, surf_type, phi0_file

def read_electric_field(param, filename):
    """
    It reads the information about the incident electric field.

    Arguments
    ---------
    param        : class, parameters related to the surface.
    filename     : name of the file that contains the infromation of the incident
                   electric field. (filname.config)

    Returns
    -------
    electric_field: float, electric field intensity, it is in the 'z'
                          direction, '-' indicates '-z'.
    wavelength   : float, wavelength of the incident electric field.
    """

    electric_field = 0
    wavelength = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()

        if len(line)>0:
            if line[0] == 'WAVE':
                electric_field = param.REAL((line[1]))
                wavelength    = param.REAL((line[2]))

    return electric_field, wavelength
