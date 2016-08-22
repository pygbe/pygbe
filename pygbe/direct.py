import numpy
import numba

#@numba.jit(cache=True)
def norm(x):
    return numpy.sqrt(numpy.sum(x**2))


@numba.jit()
def cross(a, b):

    c = numpy.array([a[1]*b[2] - a[2]*b[1],
                        a[2]*b[0] - a[0]*b[2],
                        a[0]*b[1] - a[1]*b[0]])
    return c


#@numba.jit(cache=True)
def dot(a, b):
    return numpy.dot(a, b)


#@numba.jit(cache=True)
def line_int(PHI_K, PHI_V, z, x, v1, v2, kappa, xk, wk, K, LorY):
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

    return PHI_V, PHI_K


#@numba.jit(cache=True)
def int_side(PHI_K, PHI_V, v1, v2, p, kappa, xk, wk, K, LorY):
    v21 = v2 - v1
    l21 = norm(v21)
    v21u = 1./l21 * v21
    unit = numpy.array([0., 0., 1.])
    orthog = cross(unit, v21u)

    alpha = dot(v21, v1) / (l21**2)

    #x, y, z, alpha, sign
    rorthog = -1 * alpha * v21 + v1

    d_toedge = norm(rorthog)
    #a, x
    side_vec = cross(v21, -1 * v1)

    rotate_vert = numpy.vstack((orthog, v21u, unit))

    v1new = rotate_vert @ v1

    if v1new.ravel()[0] < 0:
        v21u *= -1
        orthog += -1
        rotate_vert *= -1
        rotate_vert.ravel()[8] = 1
        v1new = rotate_vert @ v1

    v2new = rotate_vert @ v2
    rorthognew = rotate_vert @ rorthog


    if (v1new[1] > 0 and v2new[1] < 0) or (v1new[1] < 0 and v2new[1] > 0):
        PHI1_K, PHI1_V, PHI2_K, PHI2_V = 0, 0, 0, 0
        PHI1_K, PHI1_V = line_int(PHI1_K, PHI1_V, p, v1new[0], 0, v1new[1], kappa, xk, wk, K, LorY)
        PHI2_K, PHI2_V = line_int(PHI2_K, PHI2_V, p, v1new[0], v2new[1], 0, kappa, xk, wk, K, LorY)

        PHI_K += PHI1_K + PHI2_K
        PHI_V += PHI1_V + PHI2_V
    else:
        PHI_Kaux, PHI_Vaux = 0, 0
        PHI_Kaux, PHI_Vaux = line_int(PHI_Kaux, PHI_Vaux, p, v1new[0], v1new[1], v2new[1], kappa, xk, wk, K, LorY)

        PHI_K -= PHI_Kaux
        PHI_V -= PHI_Vaux

    return PHI_K, PHI_V


#@numba.jit(cache=True)
def sa(PHI_K, PHI_V, y, x, kappa, same, K_diag, V_diag, LorY, xk, xkSize, wk):
    x_panel = x[:3] - y[:3]
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

    panel0_final[:2] = panel0_plane[:2] - x_plane[:2]
    panel1_final[:2] = panel1_plane[:2] - x_plane[:2]
    panel2_final[:2] = panel2_plane[:2] - x_plane[:2]

    PHI_K, PHI_V = int_side(PHI_K, PHI_V, panel0_final, panel1_final, x_plane[2],
             kappa, xk, wk, xkSize, LorY) # Side 0
    PHI_K, PHI_V = int_side(PHI_K, PHI_V, panel1_final, panel2_final, x_plane[2],
             kappa, xk, wk, xkSize, LorY) # Side 1
    PHI_K, PHI_V = int_side(PHI_K, PHI_V, panel2_final, panel0_final, x_plane[2],
             kappa, xk, wk, xkSize, LorY) # Side 2

    if same == 1:
        PHI_K += K_diag
        PHI_V += V_diag

    return PHI_K, PHI_V

#@numba.jit(cache=True)
def compute_diagonal(vl, kl, vy, ky, triangle, centers, kappa, k_diag, v_diag, xk, wk):

    for i in range(len(vl)):
        panel = triangle[i*9: i*9+9].copy()
        center = centers[3*i: 3*i+3].copy()

        PHI_K = 0
        PHI_V = 0
        LorY = 1 # Laplace
        PHI_K, PHI_V = sa(PHI_K, PHI_V, panel, center, 1e-12, 1, k_diag, v_diag, LorY, xk, len(xk), wk)
        vl[i] = PHI_V
        kl[i] = PHI_K

        PHI_K = 0
        PHI_V = 0

        LorY = 2 # Yukawa
        PHI_K, PHI_V = sa(PHI_K, PHI_V, panel, center, kappa, 1, k_diag, v_diag, LorY, xk, len(xk), wk)

        vy[i] = PHI_V
        ky[i] = PHI_K

    return vl, kl, vy, ky


