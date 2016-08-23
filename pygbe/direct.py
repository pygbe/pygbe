import numpy
import numba


@numba.njit(cache=True)
def norm(x):
    return numpy.sqrt(numpy.sum(x**2))


@numba.njit(cache=True)
def cross(a, b):

    c = numpy.array([a[1]*b[2] - a[2]*b[1],
                    a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0]])
    return c


@numba.njit(cache=True)
def line_int(z, x, v1, v2, kappa, xk, wk, LorY):
    PHI_K = 0
    PHI_V = 0
    theta1 = numpy.arctan2(v1, x)
    theta2 = numpy.arctan2(v2, x)
    dtheta = theta2 - theta1
    thetam = (theta2 + theta1) / 2

    if abs(z) < 1e-10:
        signZ = 0
    else:
        signZ = z / abs(z)

    thetak = dtheta / 2 * xk + thetam
    Rtheta = x / numpy.cos(thetak)
    R = numpy.sqrt(Rtheta**2 + z**2)
    expKr = numpy.exp(-kappa * R)
    if LorY == 2:
        if kappa > 1e-12:
            expKz = numpy.exp(-kappa * abs(z))
            PHI_V += numpy.sum(-wk * (expKr - expKz) / kappa * dtheta / 2)
            PHI_K += numpy.sum(wk * (z / R * expKr - expKz * signZ) * dtheta / 2)
        else:
            PHI_V += numpy.sum(wk * (R - abs(z)) * dtheta / 2)
            PHI_K += numpy.sum(wk * (z / R - signZ) * dtheta / 2)
    if LorY == 1:
        PHI_V += numpy.sum(wk * (R - abs(z)) * dtheta / 2)
        PHI_K += numpy.sum(wk * (z / R - signZ) * dtheta / 2)

    return PHI_K, PHI_V


@numba.njit(cache=True)
def generate_rot_matrix(a, b, c):
    rot_matrix = numpy.empty((3, 3))
    for i in range(3):
        rot_matrix[0, i] = a[i]
        rot_matrix[1, i] = b[i]
        rot_matrix[2, i] = c[i]

    return rot_matrix


@numba.njit(cache=True)
def int_side(PHI_K, PHI_V, v1, v2, p, kappa, xk, wk, LorY):
    v21 = v2 - v1
    l21 = norm(v21)
    v21u = 1./l21 * v21
    unit = numpy.array([0., 0., 1.])
    orthog = cross(unit, v21u)

    # a, x
    rotate_vert = generate_rot_matrix(orthog, v21u, unit)

    v1new = rotate_vert @ v1

    if v1new[0] < 0:
        v21u *= -1
        orthog += -1
        rotate_vert *= -1
        rotate_vert[2, 2] = 1
        v1new = rotate_vert @ v1

    v2new = rotate_vert @ v2

    if (v1new[1] > 0 and v2new[1] < 0) or (v1new[1] < 0 and v2new[1] > 0):
        PHI1_K, PHI1_V = line_int(p, v1new[0], 0, v1new[1], kappa, xk, wk, LorY)
        PHI2_K, PHI2_V = line_int(p, v1new[0], v2new[1], 0, kappa, xk, wk, LorY)

        PHI_K += PHI1_K + PHI2_K
        PHI_V += PHI1_V + PHI2_V
    else:
        PHI_Kaux, PHI_Vaux = line_int(p, v1new[0], v1new[1], v2new[1], kappa, xk, wk, LorY)

        PHI_K -= PHI_Kaux
        PHI_V -= PHI_Vaux

    return PHI_K, PHI_V


@numba.njit(cache=True)
def sa(y, x, kappa):
    x_panel = x[:3] - y[:3]
    y0_panel = numpy.zeros(3)
    y1_panel = y[3:6] - y[:3]
    y2_panel = y[6:] - y[:3]

    X = y1_panel.copy()
    Z = cross(y1_panel, y2_panel)

    X /= numpy.sqrt(numpy.sum(X**2))
    Z /= numpy.sqrt(numpy.sum(Z**2))

    Y = cross(Z, X)

    rot_matrix = generate_rot_matrix(X, Y, Z)

    panel0_plane = rot_matrix @ y0_panel
    panel1_plane = rot_matrix @ y1_panel
    panel2_plane = rot_matrix @ y2_panel
    x_plane = rot_matrix @ x_panel

    for i in range(2):
        panel0_plane[i] -= x_plane[i]
        panel1_plane[i] -= x_plane[i]
        panel2_plane[i] -= x_plane[i]

    return panel0_plane, panel1_plane, panel2_plane, x_plane


@numba.njit(cache=True)
def compute_diagonal(vl, kl, vy, ky, triangle, centers, kappa, k_diag, v_diag, xk, wk):

    for i in range(len(vl)):
        panel = triangle[i*9: i*9+9]
        center = centers[3*i: 3*i+3]

        PHI_K = 0
        PHI_V = 0
        LorY = 1  # Laplace
        panel0_final, panel1_final, panel2_final, x_plane = sa(panel, center, 1e-12)

        PHI_K, PHI_V = int_side(0, 0, panel0_final, panel1_final, x_plane[2],
                                kappa, xk, wk, LorY)  # Side 0
        PHI_K, PHI_V = int_side(PHI_K, PHI_V, panel1_final, panel2_final, x_plane[2],
                                kappa, xk, wk, LorY)  # Side 1
        PHI_K, PHI_V = int_side(PHI_K, PHI_V, panel2_final, panel0_final, x_plane[2],
                                kappa, xk, wk, LorY)  # Side 2

        # this is replacing `same == 1`
        PHI_K += k_diag
        PHI_V += v_diag

        vl[i] = PHI_V
        kl[i] = PHI_K

        PHI_K = 0
        PHI_V = 0

        LorY = 2  # Yukawa
        # was sa(y, x, kappa, same, K_diag, V_diag, LorY, xk, wk)
        panel0_final, panel1_final, panel2_final, x_plane = sa(panel, center, kappa)

        PHI_K, PHI_V = int_side(0, 0, panel0_final, panel1_final, x_plane[2],
                                kappa, xk, wk, LorY)  # Side 0
        PHI_K, PHI_V = int_side(PHI_K, PHI_V, panel1_final, panel2_final, x_plane[2],
                                kappa, xk, wk, LorY)  # Side 1
        PHI_K, PHI_V = int_side(PHI_K, PHI_V, panel2_final, panel0_final, x_plane[2],
                                kappa, xk, wk, LorY)  # Side 2

        # this is replacing `same == 1`
        PHI_K += k_diag
        PHI_V += v_diag

        vy[i] = PHI_V
        ky[i] = PHI_K

    return vl, kl, vy, ky
