import numpy
from numpy import pi
from pygbe.util.util import calculate_gamma, test_pos
from pygbe.util.util_arr import calculate_gamma as calculate_gamma_arr
from pygbe.util.util_arr import test_pos as test_pos_arr


# y: triangle vertices
# x: source point
# same: 1 if x is in triangle, 0 if not
def AI(y, x, same):

    eps = 1e-16
    L = numpy.array([y[1] - y[0], y[2] - y[1], y[0] - y[2]])

    Lu = numpy.array([L[0] / numpy.norm(L[0]), L[1] / numpy.norm(L[1]), L[2] /
                      numpy.norm(L[2])]
                     )  # Unit vectors parallel to L0, L1 and L2

    normal = numpy.cross(L[0], L[2])
    normal = normal / numpy.norm(normal)

    theta = numpy.zeros(3)
    theta[0] = numpy.arccos(numpy.dot(Lu[0], -Lu[2]))
    theta[1] = numpy.arccos(numpy.dot(Lu[1], -Lu[0]))
    theta[2] = pi - theta[0] - theta[1]

    tu = numpy.array([numpy.cross(normal, Lu[0]), numpy.cross(normal, Lu[1]),
                      numpy.cross(normal, Lu[2])]
                     )  # Unit vector in triangle plane normal to L0, L1 and L2

    etha = numpy.dot(x - y[0], normal)
    rho = numpy.array([numpy.norm(x - y[0]), numpy.norm(x - y[1]), numpy.norm(
        x - y[2])])

    pp = numpy.zeros((3))
    pp = [numpy.dot(x - y[0], Lu[0]), numpy.dot(x - y[0], Lu[1]),
          numpy.dot(x - y[0], Lu[2])
          ]  # Distance x to y0 projected in L0, L1, and L2

    p = numpy.zeros((3, 3))
    p[0] = [numpy.dot(x - y[0], Lu[0]), numpy.dot(x - y[0], Lu[1]),
            numpy.dot(x - y[0], Lu[2])
            ]  # Distance x to y0 projected in L0, L1, and L2
    p[1] = [numpy.dot(x - y[1], Lu[0]), numpy.dot(x - y[1], Lu[1]),
            numpy.dot(x - y[1], Lu[2])
            ]  # Distance x to y1 projected in L0, L1, and L2
    p[2] = [numpy.dot(x - y[2], Lu[0]), numpy.dot(x - y[2], Lu[1]),
            numpy.dot(x - y[2], Lu[2])
            ]  # Distance x to y2 projected in L0, L1, and L2
    p = -p  # Don't really know why!

    q = numpy.zeros(3)
    q[0] = numpy.dot(x - y[0], tu[0])
    q[1] = numpy.dot(x - y[1], tu[1])
    q[2] = numpy.dot(x - y[2], tu[2])

    gamma = calculate_gamma(p, q, rho, etha)

    rhop1 = numpy.array([numpy.norm(x - y[1]), numpy.norm(x - y[2]),
                         numpy.norm(x - y[0])])
    pp1 = numpy.array([p[1, 0], p[2, 1], p[0, 2]])

    chi = numpy.log(numpy.diag(p) + rho) - numpy.log(pp1 + rhop1)

    aQ = numpy.dot(y[0] - y[2], tu[0])
    bQ = numpy.dot(y[1] - y[0], Lu[0])
    cQ = numpy.dot(y[2] - y[0], Lu[0])

    theta0 = test_pos(aQ, bQ, cQ, q, p, same)

    if (etha < 1e-10): THETA = 0.5 * numpy.sum(gamma) - theta0
    else: THETA = 0.5 * numpy.sum(gamma) + theta0

    Q = numpy.dot(q, chi) - etha * THETA
    H = -THETA

    return H, Q


def AI_arr(DorN, y, x, same, E_hat):
    # y: numpy.array of triangles

    eps = 1e-16

    # Vector parallel to each side of triangle
    L = numpy.zeros((len(y), 3, 3))
    L[:, 0] = y[:, 1] - y[:, 0]
    L[:, 1] = y[:, 2] - y[:, 1]
    L[:, 2] = y[:, 0] - y[:, 2]

    # Unit vectors parallel to L[0], L[1] and L[2]
    Lu = numpy.zeros((len(y), 3, 3))
    normL = numpy.sqrt(numpy.sum(L[:, 0]**2, axis=1))
    Lu[:, 0] = L[:, 0] / numpy.transpose(normL * numpy.ones((3, len(y))))
    normL = numpy.sqrt(numpy.sum(L[:, 1]**2, axis=1))
    Lu[:, 1] = L[:, 1] / numpy.transpose(normL * numpy.ones((3, len(y))))
    normL = numpy.sqrt(numpy.sum(L[:, 2]**2, axis=1))
    Lu[:, 2] = L[:, 2] / numpy.transpose(normL * numpy.ones((3, len(y))))

    # Normal vector to panels
    normal = numpy.zeros((len(y), 3))
    normal[:] = numpy.cross(L[:, 0], L[:, 2])
    norm_normal = numpy.sqrt(numpy.sum(normal**2, axis=1))
    normal = normal / numpy.transpose(norm_normal * numpy.ones((3, len(y))))

    theta = numpy.zeros((len(y), 3))
    theta[:, 0] = numpy.arccos(numpy.sum(Lu[:, 0] * -Lu[:, 2], axis=1))
    theta[:, 1] = numpy.arccos(numpy.sum(Lu[:, 1] * -Lu[:, 0], axis=1))
    theta[:, 2] = pi - theta[:, 0] - theta[:, 1]

    # Unit vector in triangle plane normal to L0, L1 and L2
    tu = numpy.zeros((len(y), 3, 3))
    tu[:, 0] = numpy.cross(normal[:], Lu[:, 0])
    tu[:, 1] = numpy.cross(normal[:], Lu[:, 1])
    tu[:, 2] = numpy.cross(normal[:], Lu[:, 2])

    # etha = x-y[0] \cdot normal
    etha = numpy.sum((x - y[:, 0]) * normal, axis=1)

    # rho = numpy.norm(x-y[0]), numpy.norm(x-y[1]), numpy.norm(x-y[2])
    rho = numpy.zeros((len(y), 3))
    rho[:, 0] = numpy.sqrt(numpy.sum((x - y[:, 0])**2, axis=1))
    rho[:, 1] = numpy.sqrt(numpy.sum((x - y[:, 1])**2, axis=1))
    rho[:, 2] = numpy.sqrt(numpy.sum((x - y[:, 2])**2, axis=1))

    p00 = -numpy.sum((x - y[:, 0]) * Lu[:, 0], axis=1)
    p11 = -numpy.sum((x - y[:, 1]) * Lu[:, 1], axis=1)
    p22 = -numpy.sum((x - y[:, 2]) * Lu[:, 2], axis=1)
    p10 = -numpy.sum((x - y[:, 1]) * Lu[:, 0], axis=1)
    p21 = -numpy.sum((x - y[:, 2]) * Lu[:, 1], axis=1)
    p02 = -numpy.sum((x - y[:, 0]) * Lu[:, 2], axis=1)

    q = numpy.zeros((len(y), 3))
    q[:, 0] = numpy.sum((x - y[:, 0]) * tu[:, 0], axis=1)
    q[:, 1] = numpy.sum((x - y[:, 1]) * tu[:, 1], axis=1)
    q[:, 2] = numpy.sum((x - y[:, 2]) * tu[:, 2], axis=1)

    gamma = calculate_gamma_arr(p00, p11, p22, p10, p21, p02, q, rho, etha)

    rhop1 = numpy.zeros((len(y), 3))
    rhop1[:, 0] = numpy.sqrt(numpy.sum((x - y[:, 1])**2, axis=1))
    rhop1[:, 1] = numpy.sqrt(numpy.sum((x - y[:, 2])**2, axis=1))
    rhop1[:, 2] = numpy.sqrt(numpy.sum((x - y[:, 0])**2, axis=1))

    pp1 = numpy.zeros((len(y), 3))
    pp1[:, 0] = p10
    pp1[:, 1] = p21
    pp1[:, 2] = p02

    chi = numpy.log(numpy.transpose(numpy.array([p00, p11, p22])) +
                    rho) - numpy.log(pp1 + rhop1)

    aQ = numpy.sum((y[:, 0] - y[:, 2]) * tu[:, 0], axis=1)
    bQ = numpy.sum((y[:, 1] - y[:, 0]) * Lu[:, 0], axis=1)
    cQ = numpy.sum((y[:, 2] - y[:, 0]) * Lu[:, 0], axis=1)
    theta0 = test_pos_arr(aQ, bQ, cQ, q, p00, same)

    THETA = numpy.where(etha < 1e-10,
                        0.5 * numpy.sum(gamma, axis=1) - theta0,
                        0.5 * numpy.sum(gamma, axis=1) + theta0)

    G = numpy.sum(q * chi, axis=1) - etha * THETA
    dG = -THETA

    E_HAT = numpy.where(same == 1, E_hat, 1.)
    DIAG = numpy.where(same == 1, 0., 1.)  # make diagonal of dG==0
    """
    print 'chi'
    print chi 
    print 'theta0'
    print theta0
    print 'p00 p10 p11 p21 p02 p22'
    print p00, p10, p11, p21, p02, p22
    print 'q'
    print q
    print 'gamma'
    print gamma
    print 'THETA'
    print THETA
    print 'omega'
    print numpy.dot(q[0],chi[0])
    """

    if DorN == 0:
        return G * E_HAT
    elif DorN == 1:
        return dG
    else:
        return G * E_HAT, dG * DIAG


def GQ(xi, x):
    r = numpy.norm(x - xi)
    r3 = r**3
    G = 1 / r
    Hx = (xi - x)[0] / r3
    Hy = (xi - x)[1] / r3
    Hz = (xi - x)[2] / r3

    return Hx, Hy, Hz, G


def getGaussPoints(y, triangle, n):
    # y         : vertices
    # triangle  : numpy.array with indices for corresponding triangles
    # n         : Gauss points per element

    N = len(triangle)  # Number of triangles
    xi = numpy.zeros((N * n, 3))
    if n == 1:
        for i in range(N):
            M = numpy.transpose(y[triangle[i]])
            xi[i, :] = numpy.dot(M, 1 / 3. * numpy.ones(3))

    if n == 3:
        for i in range(N):
            M = numpy.transpose(y[triangle[i]])
            xi[n * i, :] = numpy.dot(M, numpy.array([0.5, 0.5, 0.]))
            xi[n * i + 1, :] = numpy.dot(M, numpy.array([0., 0.5, 0.5]))
            xi[n * i + 2, :] = numpy.dot(M, numpy.array([0.5, 0., 0.5]))

    if n == 4:
        for i in range(N):
            M = numpy.transpose(y[triangle[i]])
            xi[n * i, :] = numpy.dot(M, numpy.array([1 / 3., 1 / 3., 1 / 3.]))
            xi[n * i + 1, :] = numpy.dot(M,
                                         numpy.array([3 / 5., 1 / 5., 1 / 5.]))
            xi[n * i + 2, :] = numpy.dot(M,
                                         numpy.array([1 / 5., 3 / 5., 1 / 5.]))
            xi[n * i + 3, :] = numpy.dot(M,
                                         numpy.array([1 / 5., 1 / 5., 3 / 5.]))

    if n == 7:
        for i in range(N):
            M = numpy.transpose(y[triangle[i]])
            xi[n * i + 0, :] = numpy.dot(M,
                                         numpy.array([1 / 3., 1 / 3., 1 / 3.]))
            xi[n * i + 1, :] = numpy.dot(
                M, numpy.array([.79742699, .10128651, .10128651]))
            xi[n * i + 2, :] = numpy.dot(
                M, numpy.array([.10128651, .79742699, .10128651]))
            xi[n * i + 3, :] = numpy.dot(
                M, numpy.array([.10128651, .10128651, .79742699]))
            xi[n * i + 4, :] = numpy.dot(
                M, numpy.array([.05971587, .47014206, .47014206]))
            xi[n * i + 5, :] = numpy.dot(
                M, numpy.array([.47014206, .05971587, .47014206]))
            xi[n * i + 6, :] = numpy.dot(
                M, numpy.array([.47014206, .47014206, .05971587]))

    return xi[:, 0], xi[:, 1], xi[:, 2]
