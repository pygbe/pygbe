"""
It contains the functions to compute the cases that presents an analytical
solutions.
All functions output the analytical solution in kcal/mol
"""
import numpy
from numpy import pi
from scipy import special, linalg
from scipy.misc import factorial
from math import gamma


def an_spherical(q, xq, E_1, E_2, E_0, R, N):
    """
    It computes the analytical solution of the potential of a sphere with
    Nq charges inside.
    Took from Kirkwood (1934).

    Arguments
    ----------
    q  : array, charges.
    xq : array, positions of the charges.
    E_1: float, dielectric constant inside the sphere.
    E_2: float, dielectric constant outside the sphere.
    E_0: float, dielectric constant of vacuum.
    R  : float, radius of the sphere.
    N  : int, number of terms desired in the spherical harmonic expansion.

    Returns
    --------
    PHI: array, reaction potential.
    """

    PHI = numpy.zeros(len(q))
    for K in range(len(q)):
        rho = numpy.sqrt(numpy.sum(xq[K]**2))
        zenit = numpy.arccos(xq[K, 2] / rho)
        azim = numpy.arctan2(xq[K, 1], xq[K, 0])

        phi = 0. + 0. * 1j
        for n in range(N):
            for m in range(-n, n + 1):
                sph1 = special.sph_harm(m, n, zenit, azim)
                cons1 = rho**n / (E_1 * E_0 * R**(2 * n + 1)) * (E_1 - E_2) * (
                    n + 1) / (E_1 * n + E_2 * (n + 1))
                cons2 = 4 * pi / (2 * n + 1)

                for k in range(len(q)):
                    rho_k = numpy.sqrt(numpy.sum(xq[k]**2))
                    zenit_k = numpy.arccos(xq[k, 2] / rho_k)
                    azim_k = numpy.arctan2(xq[k, 1], xq[k, 0])
                    sph2 = numpy.conj(special.sph_harm(m, n, zenit_k, azim_k))
                    phi += cons1 * cons2 * q[K] * rho_k**n * sph1 * sph2

        PHI[K] = numpy.real(phi) / (4 * pi)

    return PHI


def get_K(x, n):
    """
    It computes the polinomials K needed for Kirkwood-1934 solutions.
    K_n(x) in Equation 4 in Kirkwood 1934.

    Arguments
    ----------
    x: float, evaluation point of K.
    n: int, number of terms desired in the expansion.

    Returns
    --------
    K: float, polinomials K.
    """

    K = 0.
    n_fact = factorial(n)
    n_fact2 = factorial(2 * n)
    for s in range(n + 1):
        K += 2**s * n_fact * factorial(2 * n - s) / (factorial(s) * n_fact2 *
                                                     factorial(n - s)) * x**s

    return K


def an_P(q, xq, E_1, E_2, R, kappa, a, N):
    """
    It computes the solvation energy according to Kirkwood-1934.

    Arguments
    ----------
    q    : array, charges.
    xq   : array, positions of the charges.
    E_1  : float, dielectric constant inside the sphere.
    E_2  : float, dielectric constant outside the sphere.
    R    : float, radius of the sphere.
    kappa: float, reciprocal of Debye length.
    a    : float, radius of the Stern Layer.
    N    : int, number of terms desired in the polinomial expansion.

    Returns
    --------
    E_P  : float, solvation energy.
    """

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    PHI = numpy.zeros(len(q))
    for K in range(len(q)):
        rho = numpy.sqrt(numpy.sum(xq[K]**2))
        zenit = numpy.arccos(xq[K, 2] / rho)
        azim = numpy.arctan2(xq[K, 1], xq[K, 0])

        phi = 0. + 0. * 1j
        for n in range(N):
            for m in range(-n, n + 1):
                P1 = special.lpmv(numpy.abs(m), n, numpy.cos(zenit))

                Enm = 0.
                for k in range(len(q)):
                    rho_k = numpy.sqrt(numpy.sum(xq[k]**2))
                    zenit_k = numpy.arccos(xq[k, 2] / rho_k)
                    azim_k = numpy.arctan2(xq[k, 1], xq[k, 0])
                    P2 = special.lpmv(numpy.abs(m), n, numpy.cos(zenit_k))

                    Enm += q[k] * rho_k**n * factorial(n - numpy.abs(
                        m)) / factorial(n + numpy.abs(m)) * P2 * numpy.exp(
                            -1j * m * azim_k)

                C2 = (kappa * a)**2 * get_K(kappa * a, n - 1) / (
                    get_K(kappa * a, n + 1) + n * (E_2 - E_1) / (
                        (n + 1) * E_2 + n * E_1) * (R / a)**(2 * n + 1) *
                    (kappa * a)**2 * get_K(kappa * a, n - 1) / ((2 * n - 1) *
                                                                (2 * n + 1)))
                C1 = Enm / (E_2 * E_0 * a**
                            (2 * n + 1)) * (2 * n + 1) / (2 * n - 1) * (E_2 / (
                                (n + 1) * E_2 + n * E_1))**2

                if n == 0 and m == 0:
                    Bnm = Enm / (E_0 * R) * (
                        1 / E_2 - 1 / E_1) - Enm * kappa * a / (
                            E_0 * E_2 * a * (1 + kappa * a))
                else:
                    Bnm = 1. / (E_1 * E_0 * R**(2 * n + 1)) * (E_1 - E_2) * (
                        n + 1) / (E_1 * n + E_2 * (n + 1)) * Enm - C1 * C2

                phi += Bnm * rho**n * P1 * numpy.exp(1j * m * azim)

        PHI[K] = numpy.real(phi) / (4 * pi)

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J)
    E_P = 0.5 * C0 * numpy.sum(q * PHI)

    return E_P



def two_sphere(a, R, kappa, E_1, E_2, q):
    """
    It computes the analytical solution of a spherical surface and a spherical
    molecule with a center charge, both of radius R.
    Follows Cooper&Barba 2016

    Arguments
    ----------
    a    : float, center to center distance.
    R    : float, radius of surface and molecule.
    kappa: float, reciprocal of Debye length.
    E_1  : float, dielectric constant inside the sphere.
    E_2  : float, dielectric constant outside the sphere.
    q    : float, number of qe to be asigned to the charge.

    Returns
    --------
    Einter  : float, interaction energy.
    E1sphere: float, solvation energy of one sphere.
    E2sphere: float, solvation energy of two spheres together.

    Note:
    Einter should match (E2sphere - 2xE1sphere)
    """

    N = 20  # Number of terms in expansion.

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * a)
    K1p = index / (kappa * a) * K1[0:-1] - K1[1:]

    k1 = special.kv(index, kappa * a) * numpy.sqrt(pi / (2 * kappa * a))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * a)**(3 / 2.)) * special.kv(
        index, kappa * a) + numpy.sqrt(pi / (2 * kappa * a)) * K1p

    I1 = special.iv(index2, kappa * a)
    I1p = index / (kappa * a) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * a) * numpy.sqrt(pi / (2 * kappa * a))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * a)**(3 / 2.)) * special.iv(
        index, kappa * a) + numpy.sqrt(pi / (2 * kappa * a)) * I1p

    B = numpy.zeros((N, N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    M = numpy.zeros((N, N), float)
    E_hat = E_1 / E_2
    for i in range(N):
        for j in range(N):
            M[i, j] = (2 * i + 1) * B[i, j] * (
                kappa * i1p[i] - E_hat * i * i1[i] / a)
            if i == j:
                M[i, j] += kappa * k1p[i] - E_hat * i * k1[i] / a

    RHS = numpy.zeros(N)
    RHS[0] = -E_hat * q / (4 * pi * E_1 * a * a)

    a_coeff = linalg.solve(M, RHS)

    a0 = a_coeff[0]
    a0_inf = -E_hat * q / (4 * pi * E_1 * a * a) * 1 / (kappa * k1p[0])

    phi_2 = a0 * k1[0] + i1[0] * numpy.sum(a_coeff * B[:, 0]) - q / (4 * pi *
                                                                     E_1 * a)
    phi_1 = a0_inf * k1[0] - q / (4 * pi * E_1 * a)
    phi_inter = phi_2 - phi_1

    CC0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)

    Einter = 0.5 * CC0 * q * phi_inter
    E1sphere = 0.5 * CC0 * q * phi_1
    E2sphere = 0.5 * CC0 * q * phi_2

    return Einter, E1sphere, E2sphere


def constant_potential_single_point(phi0, a, r, kappa):
    """
    It computes the potential in a point 'r' due to a spherical surface
    with constant potential phi0, immersed in water. Solution to the
    Poisson-Boltzmann problem.

    Arguments
    ----------
    phi0 : float, constant potential on the surface of the sphere.
    a    : float, radius of the sphere.
    r    : float, distance from the center of the sphere to the evaluation
                  point.
    kappa: float, reciprocal of Debye length.

    Returns
    --------
    phi  : float, potential.
    """

    phi = a / r * phi0 * numpy.exp(kappa * (a - r))

    return phi


def constant_charge_single_point(sigma0, a, r, kappa, epsilon):
    """
    It computes the potential in a point 'r' due to a spherical surface
    with constant charge sigma0 immersed in water. Solution to the
    Poisson-Boltzmann problem. .

    Arguments
    ----------
    sigma0 : float, constant charge on the surface of the sphere.
    a      : float, radius of the sphere.
    r      : float, distance from the center of the sphere to the evaluation
                    point.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    phi  : float, potential.
    """

    dphi0 = -sigma0 / epsilon
    phi = -dphi0 * a * a / (1 + kappa * a) * numpy.exp(kappa * (a - r)) / r

    return phi


def constant_potential_single_charge(phi0, radius, kappa, epsilon):
    """
    It computes the surface charge of a sphere at constant potential, immersed
    in water.

    Arguments
    ----------
    phi0   : float, constant potential on the surface of the sphere.
    radius : float, radius of the sphere.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant .

    Returns
    --------
    sigma  : float, surface charge.
    """

    dphi = -phi0 * ((1. + kappa * radius) / radius)
    sigma = -epsilon * dphi  # Surface charge

    return sigma


def constant_charge_single_potential(sigma0, radius, kappa, epsilon):
    """
    It computes the surface potential on a sphere at constant charged, immersed
    in water.

    Arguments
    ----------
    sigma0 : float, constant charge on the surface of the sphere.
    radius : float, radius of the sphere.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    phi  : float, potential.
    """

    dphi = -sigma0 / epsilon
    phi = -dphi * radius / (1. + kappa * radius)  # Surface potential

    return phi



def constant_potential_twosphere(phi01, phi02, r1, r2, R, kappa, epsilon):
    """
    It computes the solvation energy of two spheres at constant potential,
    immersed in water.

    Arguments
    ----------
    phi01  : float, constant potential on the surface of the sphere 1.
    phi02  : float, constant potential on the surface of the sphere 2.
    r1     : float, radius of sphere 1.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    E_solv  : float, solvation energy.
    """


    kT = 4.1419464e-21  # at 300K
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184
    C0 = kT / qe

    phi01 /= C0
    phi02 /= C0

    k1 = special.kv(0.5, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k2 = special.kv(0.5, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    B00 = special.kv(0.5, kappa * R) * numpy.sqrt(pi / (2 * kappa * R))
    #    k1 = special.kv(0.5,kappa*r1)*numpy.sqrt(2/(pi*kappa*r1))
    #    k2 = special.kv(0.5,kappa*r2)*numpy.sqrt(2/(pi*kappa*r2))
    #    B00 = special.kv(0.5,kappa*R)*numpy.sqrt(2/(pi*kappa*R))

    i1 = special.iv(0.5, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    i2 = special.iv(0.5, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))

    a0 = (phi02 * B00 * i1 - phi01 * k2) / (B00 * B00 * i2 * i1 - k1 * k2)
    b0 = (phi02 * k1 - phi01 * B00 * i2) / (k2 * k1 - B00 * B00 * i1 * i2)

    U1 = 2 * pi * phi01 * (phi01 * numpy.exp(kappa * r1) * (kappa * r1) *
                           (kappa * r1) / numpy.sinh(kappa * r1) - pi * a0 /
                           (2 * i1))
    U2 = 2 * pi * phi02 * (phi02 * numpy.exp(kappa * r2) * (kappa * r2) *
                           (kappa * r2) / numpy.sinh(kappa * r2) - pi * b0 /
                           (2 * i2))

    print('U1: {}'.format(U1))
    print('U2: {}'.format(U2))
    print('E: {}'.format(U1 + U2))
    C1 = C0 * C0 * epsilon / kappa
    u1 = U1 * C1
    u2 = U2 * C1

    CC0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)

    E_solv = CC0 * (u1 + u2)

    return E_solv


def constant_potential_twosphere_2(phi01, phi02, r1, r2, R, kappa, epsilon):
    """
    It computes the solvation energy of two spheres at constant potential,
    immersed in water.

    Arguments
    ----------
    phi01  : float, constant potential on the surface of the sphere 1.
    phi02  : float, constant potential on the surface of the sphere 2.
    r1     : float, radius of sphere 1.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    E_solv  : float, solvation energy.
    """

    kT = 4.1419464e-21  # at 300K
    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184
    h = R - r1 - r2
    #    E_inter = r1*r2*epsilon/(4*R) * ( (phi01+phi02)**2 * log(1+numpy.exp(-kappa*h)) + (phi01-phi02)**2*log(1-numpy.exp(-kappa*h)) )
    #    E_inter = epsilon*r1*phi01**2/2 * log(1+numpy.exp(-kappa*h))
    E_solv = epsilon * r1 * r2 * (phi01**2 + phi02**2) / (4 * (r1 + r2)) * (
        (2 * phi01 * phi02) / (phi01**2 + phi02**2) * log(
            (1 + numpy.exp(-kappa * h)) /
            (1 - numpy.exp(-kappa * h))) + log(1 - numpy.exp(-2 * kappa * h)))

    CC0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    E_solv *= CC0
    return E_solv


def constant_potential_single_energy(phi0, r1, kappa, epsilon):
    """
    It computes the total energy of a single sphere at constant potential,
    inmmersed in water.

    Arguments
    ----------
    phi0   : float, constant potential on the surface of the sphere.
    r1     : float, radius of sphere.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    E      : float, total energy.
    """

    N = 1  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * r1)
    K1p = index / (kappa * r1) * K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.kv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * K1p

    a0_inf = phi0 / k1[0]
    U1_inf = a0_inf * k1p[0]

    C1 = 2 * pi * kappa * phi0 * r1 * r1 * epsilon
    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    E = C0 * C1 * U1_inf

    return E


def constant_charge_single_energy(sigma0, r1, kappa, epsilon):
    """
    It computes the total energy of a single sphere at constant charge,
    inmmersed in water.

    Arguments
    ----------
    sigma0 : float, constant charge on the surface of the sphere.
    r1     : float, radius of sphere.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    E      : float, total energy.
    """

    N = 20  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * r1)
    K1p = index / (kappa * r1) * K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.kv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * K1p

    a0_inf = -sigma0 / (epsilon * kappa * k1p[0])

    U1_inf = a0_inf * k1[0]

    C1 = 2 * pi * sigma0 * r1 * r1
    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    E = C0 * C1 * U1_inf

    return E


def constant_potential_twosphere_dissimilar(phi01, phi02, r1, r2, R, kappa,
                                            epsilon):
    """
    It computes the interaction energy for dissimilar spheres at constant
    potential, immersed in water.

    Arguments
    ----------
    phi01  : float, constant potential on the surface of the sphere 1.
    phi02  : float, constant potential on the surface of the sphere 2.
    r1     : float, radius of sphere 1.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    E_inter: float, interaction energy.
    """

    N = 20  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * r1)
    K1p = index / (kappa * r1) * K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.kv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * K1p

    K2 = special.kv(index2, kappa * r2)
    K2p = index / (kappa * r2) * K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    k2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.kv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * K2p

    I1 = special.iv(index2, kappa * r1)
    I1p = index / (kappa * r1) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.iv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * I1p

    I2 = special.iv(index2, kappa * r2)
    I2p = index / (kappa * r2) * I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    i2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.iv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * I2p

    B = numpy.zeros((N, N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    M = numpy.zeros((2 * N, 2 * N), float)
    for j in range(N):
        for n in range(N):
            M[j, n + N] = (2 * j + 1) * B[j, n] * i1[j] / k2[n]
            M[j + N, n] = (2 * j + 1) * B[j, n] * i2[j] / k1[n]
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1

    RHS = numpy.zeros(2 * N)
    RHS[0] = phi01
    RHS[N] = phi02

    coeff = linalg.solve(M, RHS)

    a = coeff[0:N] / k1
    b = coeff[N:2 * N] / k2

    a0 = a[0]
    a0_inf = phi01 / k1[0]
    b0 = b[0]
    b0_inf = phi02 / k2[0]

    U1_inf = a0_inf * k1p[0]
    U1_h = a0 * k1p[0] + i1p[0] * numpy.sum(b * B[:, 0])

    U2_inf = b0_inf * k2p[0]
    U2_h = b0 * k2p[0] + i2p[0] * numpy.sum(a * B[:, 0])

    C1 = 2 * pi * kappa * phi01 * r1 * r1 * epsilon
    C2 = 2 * pi * kappa * phi02 * r2 * r2 * epsilon
    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    E_inter = C0 * (C1 * (U1_h - U1_inf) + C2 * (U2_h - U2_inf))

    return E_inter


def constant_charge_twosphere_dissimilar(sigma01, sigma02, r1, r2, R, kappa,
                                         epsilon):
    """
    It computes the interaction energy between two dissimilar spheres at
    constant charge, immersed in water.

    Arguments
    ----------
    sigma01: float, constant charge on the surface of the sphere 1.
    sigma02: float, constant charge on the surface of the sphere 2.
    r1     : float, radius of sphere 1.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    E_inter: float, interaction energy.
    """

    N = 20  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * r1)
    K1p = index / (kappa * r1) * K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.kv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * K1p

    K2 = special.kv(index2, kappa * r2)
    K2p = index / (kappa * r2) * K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    k2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.kv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * K2p

    I1 = special.iv(index2, kappa * r1)
    I1p = index / (kappa * r1) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.iv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * I1p

    I2 = special.iv(index2, kappa * r2)
    I2p = index / (kappa * r2) * I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    i2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.iv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * I2p

    B = numpy.zeros((N, N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    M = numpy.zeros((2 * N, 2 * N), float)
    for j in range(N):
        for n in range(N):
            M[j, n + N] = (2 * j + 1) * B[j, n] * r1 * i1p[j] / (r2 * k2p[n])
            M[j + N, n] = (2 * j + 1) * B[j, n] * r2 * i2p[j] / (r1 * k1p[n])
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1

    RHS = numpy.zeros(2 * N)
    RHS[0] = sigma01 * r1 / epsilon
    RHS[N] = sigma02 * r2 / epsilon

    coeff = linalg.solve(M, RHS)

    a = coeff[0:N] / (-r1 * kappa * k1p)
    b = coeff[N:2 * N] / (-r2 * kappa * k2p)

    a0 = a[0]
    a0_inf = -sigma01 / (epsilon * kappa * k1p[0])
    b0 = b[0]
    b0_inf = -sigma02 / (epsilon * kappa * k2p[0])

    U1_inf = a0_inf * k1[0]
    U1_h = a0 * k1[0] + i1[0] * numpy.sum(b * B[:, 0])

    U2_inf = b0_inf * k2[0]
    U2_h = b0 * k2[0] + i2[0] * numpy.sum(a * B[:, 0])

    C1 = 2 * pi * sigma01 * r1 * r1
    C2 = 2 * pi * sigma02 * r2 * r2
    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    E_inter = C0 * (C1 * (U1_h - U1_inf) + C2 * (U2_h - U2_inf))

    return E_inter


def molecule_constant_potential(q, phi02, r1, r2, R, kappa, E_1, E_2):
    """
    It computes the interaction energy between a molecule (sphere with
    point-charge in the center) and a sphere at constant potential, immersed
    in water.

    Arguments
    ----------
    q      : float, number of qe to be asigned to the charge.
    phi02  : float, constant potential on the surface of the sphere 2.
    r1     : float, radius of sphere 1, i.e the molecule.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    E_1    : float, dielectric constant inside the sphere/molecule.
    E_2    : float, dielectric constant outside the sphere/molecule.

    Returns
    --------
    E_inter: float, interaction energy.
    """

    N = 20  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * r1)
    K1p = index / (kappa * r1) * K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.kv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * K1p

    K2 = special.kv(index2, kappa * r2)
    K2p = index / (kappa * r2) * K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    k2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.kv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * K2p

    I1 = special.iv(index2, kappa * r1)
    I1p = index / (kappa * r1) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.iv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * I1p

    I2 = special.iv(index2, kappa * r2)
    I2p = index / (kappa * r2) * I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    i2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.iv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * I2p

    B = numpy.zeros((N, N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    E_hat = E_1 / E_2
    M = numpy.zeros((2 * N, 2 * N), float)
    for j in range(N):
        for n in range(N):
            M[j, n + N] = (2 * j + 1) * B[j, n] * (
                kappa * i1p[j] / k2[n] - E_hat * j / r1 * i1[j] / k2[n])
            M[j + N, n] = (2 * j + 1) * B[j, n] * i2[j] * 1 / (
                kappa * k1p[n] - E_hat * n / r1 * k1[n])
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1

    RHS = numpy.zeros(2 * N)
    RHS[0] = -E_hat * q / (4 * pi * E_1 * r1 * r1)
    RHS[N] = phi02

    coeff = linalg.solve(M, RHS)

    a = coeff[0:N] / (kappa * k1p - E_hat * numpy.arange(N) / r1 * k1)
    b = coeff[N:2 * N] / k2

    a0 = a[0]
    a0_inf = -E_hat * q / (4 * pi * E_1 * r1 * r1) * 1 / (kappa * k1p[0])
    b0 = b[0]
    b0_inf = phi02 / k2[0]

    phi_inf = a0_inf * k1[0] - q / (4 * pi * E_1 * r1)
    phi_h = a0 * k1[0] + i1[0] * numpy.sum(b * B[:, 0]) - q / (4 * pi * E_1 *
                                                               r1)
    phi_inter = phi_h - phi_inf

    U_inf = b0_inf * k2p[0]
    U_h = b0 * k2p[0] + i2p[0] * numpy.sum(a * B[:, 0])
    U_inter = U_h - U_inf

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = q * 0.5
    C2 = 2 * pi * kappa * phi02 * r2 * r2 * E_2
    E_inter = C0 * (C1 * phi_inter + C2 * U_inter)

    return E_inter


def molecule_constant_charge(q, sigma02, r1, r2, R, kappa, E_1, E_2):
    """
    It computes the interaction energy between a molecule (sphere with
    point-charge in the center) and a sphere at constant charge, immersed
    in water.

    Arguments
    ----------
    q      : float, number of qe to be asigned to the charge.
    sigma02: float, constant charge on the surface of the sphere 2.
    r1     : float, radius of sphere 1, i.e the molecule.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    E_1    : float, dielectric constant inside the sphere/molecule.
    E_2    : float, dielectric constant outside the sphere/molecule.

    Returns
    --------
    E_inter: float, interaction energy.
    """

    N = 20  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * r1)
    K1p = index / (kappa * r1) * K1[0:-1] - K1[1:]
    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.kv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * K1p

    K2 = special.kv(index2, kappa * r2)
    K2p = index / (kappa * r2) * K2[0:-1] - K2[1:]
    k2 = special.kv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    k2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.kv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * K2p

    I1 = special.iv(index2, kappa * r1)
    I1p = index / (kappa * r1) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r1)**(3 / 2.)) * special.iv(
        index, kappa * r1) + numpy.sqrt(pi / (2 * kappa * r1)) * I1p

    I2 = special.iv(index2, kappa * r2)
    I2p = index / (kappa * r2) * I2[0:-1] + I2[1:]
    i2 = special.iv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))
    i2p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * r2)**(3 / 2.)) * special.iv(
        index, kappa * r2) + numpy.sqrt(pi / (2 * kappa * r2)) * I2p

    B = numpy.zeros((N, N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    E_hat = E_1 / E_2
    M = numpy.zeros((2 * N, 2 * N), float)
    for j in range(N):
        for n in range(N):
            M[j, n + N] = (2 * j + 1) * B[j, n] * (
                i1p[j] / k2p[n] - E_hat * j / r1 * i1[j] / (kappa * k2p[n]))
            M[j + N, n] = (2 * j + 1) * B[j, n] * i2p[j] * kappa * 1 / (
                kappa * k1p[n] - E_hat * n / r1 * k1[n])
            if n == j:
                M[j, n] = 1
                M[j + N, n + N] = 1

    RHS = numpy.zeros(2 * N)
    RHS[0] = -E_hat * q / (4 * pi * E_1 * r1 * r1)
    RHS[N] = -sigma02 / E_2

    coeff = linalg.solve(M, RHS)

    a = coeff[0:N] / (kappa * k1p - E_hat * numpy.arange(N) / r1 * k1)
    b = coeff[N:2 * N] / (kappa * k2p)

    a0 = a[0]
    a0_inf = -E_hat * q / (4 * pi * E_1 * r1 * r1) * 1 / (kappa * k1p[0])
    b0 = b[0]
    b0_inf = -sigma02 / (E_2 * kappa * k2p[0])

    phi_inf = a0_inf * k1[0] - q / (4 * pi * E_1 * r1)
    phi_h = a0 * k1[0] + i1[0] * numpy.sum(b * B[:, 0]) - q / (4 * pi * E_1 *
                                                               r1)
    phi_inter = phi_h - phi_inf

    U_inf = b0_inf * k2[0]
    U_h = b0 * k2[0] + i2[0] * numpy.sum(a * B[:, 0])
    U_inter = U_h - U_inf

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = q * 0.5
    C2 = 2 * pi * sigma02 * r2 * r2
    E_inter = C0 * (C1 * phi_inter + C2 * U_inter)

    return E_inter


def constant_potential_twosphere_identical(phi01, phi02, r1, r2, R, kappa,
                                           epsilon):
    """
    It computes the interaction energy for two spheres at constants surface
    potential, according to Carnie&Chan-1993.

    Arguments
    ----------
    phi01  : float, constant potential on the surface of the sphere 1.
    phi02  : float, constant potential on the surface of the sphere 2.
    r1     : float, radius of sphere 1.
    r2     : float, radius of sphere 2.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Note:
         Even though it admits phi01 and phi02, they should be identical; and
         the same is applicable to r1 and r2.

    Returns
    --------
    E_inter: float, interaction energy.
    """

    #   From Carnie+Chan 1993

    N = 20  # Number of terms in expansion

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index = numpy.arange(N, dtype=float) + 0.5

    k1 = special.kv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    k2 = special.kv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))

    i1 = special.iv(index, kappa * r1) * numpy.sqrt(pi / (2 * kappa * r1))
    i2 = special.iv(index, kappa * r2) * numpy.sqrt(pi / (2 * kappa * r2))

    B = numpy.zeros((N, N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    M = numpy.zeros((N, N), float)
    for i in range(N):
        for j in range(N):
            M[i, j] = (2 * i + 1) * B[i, j] * i1[i]
            if i == j:
                M[i, j] += k1[i]

    RHS = numpy.zeros(N)
    RHS[0] = phi01

    a = linalg.solve(M, RHS)

    a0 = a[0]

    U = 4 * pi * (-pi / 2 * a0 / phi01 * 1 / numpy.sinh(kappa * r1) + kappa *
                  r1 + kappa * r1 / numpy.tanh(kappa * r1))

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)
    C1 = r1 * epsilon * phi01 * phi01
    E_inter = U * C1 * C0

    return E_inter


def constant_charge_twosphere_identical(sigma, a, R, kappa, epsilon):
    """
    It computes the interaction energy for two spheres at constants surface
    charge, according to Carnie&Chan-1993.

    Arguments
    ----------
    sigma  : float, constant charge on the surface of the spheres.
    a      : float, radius of spheres.
    R      : float, distance center to center.
    kappa  : float, reciprocal of Debye length.
    epsilon: float, water dielectric constant.

    Returns
    --------
    E_inter: float, interaction energy.
    """

    #   From Carnie+Chan 1993

    N = 10  # Number of terms in expansion
    E_p = 0  # Permitivitty inside sphere

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    index2 = numpy.arange(N + 1, dtype=float) + 0.5
    index = index2[0:-1]

    K1 = special.kv(index2, kappa * a)
    K1p = index / (kappa * a) * K1[0:-1] - K1[1:]

    k1 = special.kv(index, kappa * a) * numpy.sqrt(pi / (2 * kappa * a))
    k1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * a)**(3 / 2.)) * special.kv(
        index, kappa * a) + numpy.sqrt(pi / (2 * kappa * a)) * K1p

    I1 = special.iv(index2, kappa * a)
    I1p = index / (kappa * a) * I1[0:-1] + I1[1:]
    i1 = special.iv(index, kappa * a) * numpy.sqrt(pi / (2 * kappa * a))
    i1p = -numpy.sqrt(pi / 2) * 1 / (2 * (kappa * a)**(3 / 2.)) * special.iv(
        index, kappa * a) + numpy.sqrt(pi / (2 * kappa * a)) * I1p

    B = numpy.zeros((N, N), dtype=float)

    for n in range(N):
        for m in range(N):
            for nu in range(N):
                if n >= nu and m >= nu:
                    g1 = gamma(n - nu + 0.5)
                    g2 = gamma(m - nu + 0.5)
                    g3 = gamma(nu + 0.5)
                    g4 = gamma(m + n - nu + 1.5)
                    f1 = factorial(n + m - nu)
                    f2 = factorial(n - nu)
                    f3 = factorial(m - nu)
                    f4 = factorial(nu)
                    Anm = g1 * g2 * g3 * f1 * (n + m - 2 * nu + 0.5) / (
                        pi * g4 * f2 * f3 * f4)
                    kB = special.kv(n + m - 2 * nu + 0.5, kappa *
                                    R) * numpy.sqrt(pi / (2 * kappa * R))
                    B[n, m] += Anm * kB

    M = numpy.zeros((N, N), float)
    for i in range(N):
        for j in range(N):
            M[i, j] = (2 * i + 1) * B[i, j] * (
                E_p / epsilon * i * i1[i] - a * kappa * i1p[i])
            if i == j:
                M[i, j] += (E_p / epsilon * i * k1[i] - a * kappa * k1p[i])

    RHS = numpy.zeros(N)
    RHS[0] = a * sigma / epsilon

    a_coeff = linalg.solve(M, RHS)

    a0 = a_coeff[0]

    C0 = a * sigma / epsilon
    CC0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J * E_0)

    E_inter = 4 * pi * a * epsilon * C0 * C0 * CC0(pi * a0 / (2 * C0 * (
        kappa * a * numpy.cosh(kappa * a) - numpy.sinh(kappa * a))) - 1 / (
            1 + kappa * a) - 1 / (kappa * a * 1 / numpy.tanh(kappa * a) - 1))

    return E_inter

def Cext_analytical(radius, wavelength, diel_out, diel_in):
    """
    Calculates the analytical solution of the extinction cross section.
    This solution is valid when the nano particle involved is a sphere. 
    
    Arguments
    ----------
    radius    : float, radius of the sphere in [nm].
    wavelength: float/array of floats, wavelength of the incident
                electric field in [nm].
    diel_out  : complex/array of complex, dielectric constant inside surface.
    diel_in   : complex/array of complex, dielectric constant inside surface. 

    Returns
    --------
    Cext_an   : float/array of floats, extinction cross section.     
    """
    wavenumber = 2 * numpy.pi * numpy.sqrt(diel_out) / wavelength
    C1 = wavenumber**2 * (diel_in / diel_out - 1) / (diel_in / diel_out + 2)
    Cext_an = 4 * numpy.pi * radius**3 / wavenumber.real * C1.imag 
    
    return Cext_an
