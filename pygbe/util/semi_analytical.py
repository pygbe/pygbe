"""
It contains the functions needed to compute the near singular integrals in
python. For big problems these functions were written in C++ and we call them
through the pycuda interface. At the end you have a commented test to compare
both types.
"""

import numpy
from pygbe.util.semi_analyticalwrap import SA_wrap_arr


def GQ_1D(K):
    """
    Gauss quadrature in 1D.

    Arguments
    ----------
    K: int, desired number of gauss points.

    Returns
    --------
    x: float, location of the gauss point.
    w: float, weights of the gauss point.
    """

    T = numpy.zeros((K, K))
    nvec = numpy.arange(1., K)
    beta = 0.5 / numpy.sqrt(1 - 1 / (2 * nvec)**2)
    T = numpy.diag(beta, 1) + numpy.diag(beta, -1)
    d, v = numpy.linalg.eig(T)
    w = 2 * v[0]**2
    x = d

    return x, w


def lineInt(z, x, v1, v2, kappa, xk, wk):
    """
    Line integral to solve the non-analytical part (integral in the angle) in
    the semi_analytical integrals needed to calculate the potentials.

    Arguments
    ----------
    z     : float, distance (height) between the plane of the triangle and the
                   collocation point.
    x     : float, position of the collocation point.
    v1    : float, low extreme integral value.
    v2    : float, high extreme integral value.
    kappa : float, reciprocal of Debye length.
    xk    : float, position of the gauss point.
    wk    : float, weight of the gauss point.

    Returns
    --------
    phi_Y : float, potential due to a Yukawa kernel.
    dphi_Y: float, normal derivative of potential due to a Yukawa kernel.
    phi_L : float, potential due to a Laplace kernel.
    dphi_L: float, normal derivative of potential due to a Laplace kernel.
    """


    theta1 = numpy.arctan2(v1, x)
    theta2 = numpy.arctan2(v2, x)

    dtheta = theta2 - theta1

    absZ = abs(z)
    if absZ < 1e-10: signZ = 0
    else: signZ = z / absZ

    dtheta = theta2 - theta1
    thetam = (theta2 + theta1) / 2.

    thetak = dtheta / 2 * xk + thetam
    Rtheta = x / numpy.cos(thetak)
    dy = x * numpy.tan(thetak)
    R = numpy.sqrt(Rtheta**2 + z**2)

    phi_Y = numpy.sum(-wk * (exp(-kappa * R) - exp(-kappa * absZ)) / kappa)
    dphi_Y = -numpy.sum(wk *
                        (z / R * exp(-kappa * R) - exp(-kappa * absZ) * signZ))

    phi_L = numpy.sum(wk * (R - absZ))
    dphi_L = -numpy.sum(wk * (z / R - signZ))

    phi_Y *= dtheta / 2
    dphi_Y *= dtheta / 2
    phi_L *= dtheta / 2
    dphi_L *= dtheta / 2

    return phi_Y, dphi_Y, phi_L, dphi_L


def intSide(v1, v2, p, kappa, xk, wk):
    """
    It solves the integral line over one side of the triangle .

    Arguments
    ----------
    v1    : float, low extreme integral value.
    v2    : float, high extreme integral value.
    p     : float, distance (height) between the plane of the triangle and the
                   collocation point.
    kappa : float, reciprocal of Debye length.
    xk    : float, position of the gauss point.
    wk    : float, weight of the gauss point.

    Returns
    --------
    phi_Y : float, potential due to a Yukawa kernel.
    dphi_Y: float, normal derivative of potential due to a Yukawa kernel.
    phi_L : float, potential due to a Laplace kernel.
    dphi_L: float, normal derivative of potential due to a Laplace kernel.
    """

    v21 = v2 - v1
    L21 = numpy.linalg.norm(v21)
    v21u = v21 / L21
    orthog = numpy.cross(numpy.array([0, 0, 1]), v21u)

    alpha = -numpy.dot(v21, v1) / L21**2

    rOrthog = v1 + alpha * v21
    d_toEdge = numpy.linalg.norm(rOrthog)
    side_vec = numpy.cross(v21, -v1)

    rotateToVertLine = numpy.zeros((3, 3))
    rotateToVertLine[:, 0] = orthog
    rotateToVertLine[:, 1] = v21u
    rotateToVertLine[:, 2] = [0., 0., 1.]

    v1new = numpy.dot(rotateToVertLine, v1)

    if v1new[0] < 0:
        v21u = -v21u
        orthog = -orthog
        rotateToVertLine[:, 0] = orthog
        rotateToVertLine[:, 1] = v21u
        v1new = numpy.dot(rotateToVertLine, v1)

    v2new = numpy.dot(rotateToVertLine, v2)
    rOrthognew = numpy.dot(rotateToVertLine, rOrthog)
    x = v1new[0]

    if v1new[1] > 0 and v2new[1] < 0 or v1new[1] < 0 and v2new[1] > 0:
        phi1_Y, dphi1_Y, phi1_L, dphi1_L = lineInt(p, x, 0, v1new[1], kappa,
                                                   xk, wk)
        phi2_Y, dphi2_Y, phi2_L, dphi2_L = lineInt(p, x, v2new[1], 0, kappa,
                                                   xk, wk)

        phi_Y = phi1_Y + phi2_Y
        dphi_Y = dphi1_Y + dphi2_Y
        phi_L = phi1_L + phi2_L
        dphi_L = dphi1_L + dphi2_L

    else:
        phi_Y, dphi_Y, phi_L, dphi_L = lineInt(p, x, v1new[1], v2new[1], kappa,
                                               xk, wk)
        phi_Y = -phi_Y
        dphi_Y = -dphi_Y
        phi_L = -phi_L
        dphi_L = -dphi_L

    return phi_Y, dphi_Y, phi_L, dphi_L


def SA_arr(y, x, kappa, same, xk, wk):
    """
    It computes the integral line for all the sides of a triangle and for all
    the collocation points.

    Arguments
    ----------
    y     : array, vertices coordinates of the triangles.
    x     : array, collocation points.
    kappa : float, reciprocal of Debye length.
    same  : int, 1 if the collocation point is in the panel of integration,
                 0 otherwise.
    xk    : float, position of the gauss point.
    wk    : float, weight of the gauss point.

    Returns
    --------
    phi_Y : float, potential due to a Yukawa kernel.
    dphi_Y: float, normal derivative of potential due to a Yukawa kernel.
    phi_L : float, potential due to a Laplace kernel.
    dphi_L: float, normal derivative of potential due to a Laplace kernel.
    """

    N = len(x)
    phi_Y = numpy.zeros(N)
    dphi_Y = numpy.zeros(N)
    phi_L = numpy.zeros(N)
    dphi_L = numpy.zeros(N)

    # Put first vertex at origin
    y_panel = y - y[0]
    x_panel = x - y[0]

    # Find panel coordinate system X: 0->1
    X = y_panel[1]
    X = X / numpy.linalg.norm(X)
    Z = numpy.cross(y_panel[1], y_panel[2])
    Z = Z / numpy.linalg.norm(Z)
    Y = numpy.cross(Z, X)

    # Rotate coordinate system to match panel plane
    rot_matrix = numpy.array([X, Y, Z])
    panel_plane = numpy.transpose(numpy.dot(rot_matrix, numpy.transpose(
        y_panel)))
    x_plane = numpy.transpose(numpy.dot(rot_matrix, numpy.transpose(x_panel)))

    for i in range(N):
        # Shift origin so it matches collocation point
        panel_final = panel_plane - numpy.array([x_plane[i, 0], x_plane[i, 1],
                                                 0])

        # Loop over sides
        for j in range(3):
            if j == 2: nextJ = 0
            else: nextJ = j + 1

            phi_Y_aux, dphi_Y_aux, phi_L_aux, dphi_L_aux = intSide(
                panel_final[j], panel_final[nextJ], x_plane[i, 2], kappa, xk,
                wk)
            phi_Y[i] += phi_Y_aux
            dphi_Y[i] += dphi_Y_aux
            phi_L[i] += phi_L_aux
            dphi_L[i] += dphi_L_aux

        if same[i] == 1:
            dphi_Y[i] = 2 * pi
            dphi_L[i] = -2 * pi

    return phi_Y, dphi_Y, phi_L, dphi_L
