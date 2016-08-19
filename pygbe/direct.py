import numpy
import numba

@numba.njit(cache=True)
def norm(x):
    return numpy.sqrt(numpy.sum(x**2))


@numba.njit(cache=True)
def cross(a, b):

    return numpy.array([a[1]*b[2] - a[2]*b[1],
                        a[2]*b[0] - a[0]*b[2],
                        a[0]*b[1] - a[1]*b[0]])


@numba.njit(cache=True)
def matvec(a, b):
    return a @ b


@numba.njit(cache=True)
def dot(a, b):
    return numpy.dot(a, b)


@numba.njit(cache=True)
def axpy(a, x, y, sign):
    return sign * a * x + y


@numba.njit(cache=True)
def ax(a, x):
    return a * x


@numba.njit(cache=True)
def line_int(z, x, v1, v2, kappa, xk, wk, K, LorY):
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

    PHI_V, PHI_K = 0, 0

    if LorY == 2 and kappa > 1e-12:
        expKz = numpy.exp(-kappa * abs(z))
        PHI_V += numpy.sum(-wk * (expKr - expKz) / kappa * dtheta / 2)
        PHI_K += numpy.sum(wk * (z / R * expKr - expKz * signZ) * dtheta / 2)
    elif LorY == 2 or LorY == 1:
        PHI_V += numpy.sum(wk * (R - abs(z)) * dtheta / 2)
        PHI_K += numpy.sum(wk * (z / R - signZ) * dtheta / 2)

    return PHI_V, PHI_K


@numba.njit(cache=True)
def int_side(PHI_K, PHI_V, v2, v1, p, kappa, xk, wk, K, LorY):
    v21 = v2 - v1
    l21 = norm(v21)
    v21u = 1./l21 * v21
    unit = numpy.array([0., 0., 1.])
    orthog = cross(unit, v21u)

    alpha = dot(v21, v1) / (L21**2)

    #a, x, y, sign
    rorthog = -1 * alpha * x + y

    d_toedge = norm(rorthog)
    #a, x
    side_vec = cross(v21, -1 * vx)

    rotate_vert = numpy.vstack((orthog, v21u, unit))

    v1new = rotate_vert @ v1

    if v1new[0] < 0:
        v21u *= -1
        orthog += -1
        rotate_vert *= -1
        rotate_vert[8] = 1
        v1new = rotate_vert @ v1

    v2new = rotate_vert @ v21u
    rorthognew = rotate_vert @ rorthog

    if (v1new[1] > 0 and v2new[1] < 0) or (v1new[1] < 0 and v2new[1] > 0):
        PHI1_K, PHI1_V = line_int(p, x, 0, v1new[1], kappa, xk, wk, K, LorY)
        PHI2_K, PHI2_V = line_int(p, x, v2new[1], 0, kappa, xk, wk, K, LorY)

        PHI_K += PHI1_K + PHI2_K
        PHI_V += PHI1_V + PHI2_V
    else:
        PHI_Kaux, PHI_Vaux = line_int(p, x, v1new[1], v2new[1], kappa, xk, wk, K, LorY)

        PHI_K -= PHI_Kaux
        PHI_V -= PHI_Vaux


@numba.njit(cache=True)
def sa(PHI_K, PHI_V, y, x, kappa, same, K_diag, V_diag, LorY, xk, xkSize, wk):
    x_panel = x - y
    y0_panel = numpy.zeros(3)
    y1_panel = y[3:6] - y[:3]
    y2_panel = y[6:] - y[:3]

    X = y1_panel.copy()
    Z = cross(y1_panel, y2_panel)

    X /= norm(X)
    Z /= norm(Z)

    Y = cross(Z, X)

    rot_matrix = numpy.vstack((X, Y, Z))

    panel0_plane = rot_matrix @ y0_panel
    panel1_plane = rot_matrix @ y1_panel
    panel2_plane = rot_matrix @ y2_panel
    x_plane = rot_matrix @ x_panel

    panel0_final = panel0_plane.copy()
    panel1_final = panel1_plane.copy()
    panel2_final = panel2_plane.copy()

    panel0_final[:3] = panel0_plane[:3] - x_plane[:3]
    panel1_final[:3] = panel1_plane[:3] - x_plane[:3]
    panel2_final[:3] = panel2_plane[:3] - x_plane[:3]

    int_side(PHI_K, PHI_V, panel0_final, panel1_final, x_plane[2],
             kappa, xk, wk, xkSize, LorY) # Side 0
    int_side(PHI_K, PHI_V, panel1_final, panel2_final, x_plane[2],
             kappa, xk, wk, xkSize, LorY) # Side 1
    int_side(PHI_K, PHI_V, panel2_final, panel0_final, x_plane[2],
             kappa, xk, wk, xkSize, LorY) # Side 2

    if same == 1:
        PHI_K += K_diag
        PHI_V += V_diag

# Need compute diagonal
## requires SA
### requires cross, norm, matvec, intSide
## intSide
### requires ax, cross, axpy, dot_prod, norm, matvec, lineint
