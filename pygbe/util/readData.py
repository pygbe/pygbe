import numpy
import os


def readVertex2(filename, REAL):
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


def readVertex(filename, REAL):
    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')
    geo_path = os.environ.get('PYGBE_GEOMETRY')
    if geo_path:
        full_path = geo_path
    X = numpy.loadtxt(os.path.join(full_path, filename), dtype=REAL)
    vertex = X[:, 0:3]

    return vertex


def readTriangle2(filename):
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


def readTriangle(filename, surf_type):
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


def readCheck(aux, REAL):
    # check if it is not reading more than one term
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


def readSurf(filename):

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
