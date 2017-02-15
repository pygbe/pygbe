import numpy
from numba import njit

@njit
def setIndex(P, i, j, k):
    I = 0
    for ii in range(i):
        for jj in range(1, P + 2 - ii):
            I += jj

    for jj in range(P + 2 - j, P + 2):
        I += jj - i

    I += k

    return I

@njit
def getIndex_arr(P, ii, jj, kk):
    indices = numpy.zeros_like(ii, dtype=numpy.int32)
    for i in range(len(ii)):
        indices[i] = setIndex(P, ii[i], jj[i], kk[i])

    return indices

@njit
def multipole_sort(K_aux, V_aux, offTar, sizeTar, offMlt, M, Md, xi, yi, zi, xc, yc, zc, index, P, kappa, Nm, LorY):
    NotImplemented

@njit
def getIndex(P, i, j, k):
    return i * (P + 1)**2 + j * (P + 1) + k

@njit
def getCoeff(a, dx, dy, dz, index, Nm, P, kappa, LorY):
    b = numpy.zeros(Nm)
    laplace, yukawa = False, False
    if LorY == 1:
        laplace = True
    elif LorY == 2:
        yukawa = True

    R = (dx**2 + dy**2 + dz**2)**.5

    I = index[getIndex(P, 1, 0, 0)]

    if laplace:
        a[0] = 1 / R

        a[I] = -dx / R**3
        a[P + 1] = -dy / R**3
        a[1] = -dz / R**3

    elif yukawa
        b[0] = numpy.exp(-kappa * R)
        a[0] = b[0] / R


        b[I]   = -kappa * (dx * a[0])  # 1,0,0
        b[P + 1] = -kappa * (dy * a[0])  # 0,1,0
        b[1]   = -kappa * (dz * a[0])  # 0,0,1

        a[i]   = -1 / r ** 2 * (kappa * dx * b[0] + dx * a[0])
        a[p + 1] = -1 / r ** 2 * (kappa * dy * b[0] + dy * a[0])
        a[1]   = -1 / r ** 2 * (kappa * dz * b[0] + dz * a[0])

    for i in range(2, P+1):
        Cb = -kappa / i
        C = 1 / (i * R**2)
        Ix = index[getIndex(P, i, 0, 0)]
        Iy = index[getIndex(P, 0, i, 0)]
        Iz = index[getIndex(P, 0, 0, i)]

        Im1x = index[getIndex(P, i - 1, 0, 0)]
        Im2x = index[getIndex(P, i - 2, 0, 0)]
        Im1y = I - (P + 2 - i)
        Im2y = Im1y - (P + 2 - i + 1)
        Im1z = Iz - 1
        Im2z = Iz - 2

        diff = numpy.array([dx, dy, dz])

        if laplace:
            a[[Ix, Iy, Iz]] = (C * (-(2 * i - 1)
                                    * diff
                                    * a[[Im1x, Im1y, Im1z]] - (i - 1)
                                    * a[[Im2x, Im2y, Im2z]]))
        elif yukawa:
            b[[Ix, Iy, Iz]] = Cb * (diff * a[[Im1x, Im1y, Im1z]]
                                    + a[[Im2x, Im2y, Im2z]])
            a[[Ix, Iy, Iz]] = C * (-kappa * (diff
                                             * b[[Im1x, Im1y, Im1z]]
                                             + b[[Im2x, Im2y, Im2z]])
                                   - (2 * i - 1) * diff
                                   * a[[Im1x, Im1y, Im1z]] - (i - 1)
                                   * a[[Im2x, Im2y, Im2z]])

    # TODO: collapse all of these

    Cb = -kappa / 2
    I = index[getIndex(P,1,1,0)]
    Im1x = P + 1
    Im1y = I - (P + 2 - 1 - 1)

    if laplace:
        a[I] = 1 / (2 * R**2) * ( -(2 * 2 - 1) * (dx * a[Im1x] + dy * a[Im1y]) )
    elif yukawa:
        b[I] = Cb * (dx * a[Im1x] + dy * a[Im1y])
        a[I] = (1/(2 * R**2)
                * ( -kappa * (dx * b[Im1x] + dy * b[Im1y])
                    - (2 * 2 - 1) * (dx * a[Im1x] + dy * a[Im1y]) ))

    I = index[getIndex(P,1,0,1)]
    Im1x = 1
    Im1z = I - 1

    if laplace:
        a[I] = 1 / (2 * R**2) * ( -(2 * 2 - 1) * (dx * a[Im1x] + dz * a[Im1z]) )
    elif yukawa:
        b[I] = Cb * (dx * a[Im1x] + dz * a[Im1z])
        a[I] = (1/(2 * R**2)
                * ( -kappa * (dx * b[Im1x] + dz * b[Im1z])
                    - (2 * 2 - 1) * (dx * a[Im1x] + dz * a[Im1z]) ))

    I = index[getIndex(P,0,1,1)]
    Im1y = I - (P + 2 - 1)
    Im1z = I - 1

    if laplace:
        a[I] = 1 / (2 * R**2) * ( -(2 * 2 - 1) * (dy * a[Im1y] + dz * a[Im1z]) )
    elif yukawa:
        b[I] = Cb * (dy * a[Im1y] + dz * a[Im1z])
        a[I] = (1/(2 * R**2)
                * ( -kappa * (dy * b[Im1y] + dz * b[Im1z])
                    - (2 * 2 - 1) * (dy * a[Im1y] + dz * a[Im1z]) ))

    # porting from line 176 of multipole.cpp
    for i in range(2, P):
        Cb = -kappa / (i + 1)
        C = 1 / ((1 + i) * R**2)

        I = index[getIndex(P, 1, i, 0)]
        Im1x = index[getIndex(P, 0, i, 0)]
        Im1y = I - (P + 2 - i - 1)
        Im2y = Im1y - (P + 2 - i)

        yukawa_1(b, i, Cb, C, dx, Im1x, dy, Im1y, Im2y, kappa)




@njit
def yukawa_1(b, i, Cb, C, d_1, Im1_1, d_2, Im1_2, Im2_2, kappa):
    b[I] = Cb * (d_1 * a[Im1_1] + d_2 * a[Im1_2] + a[Im2_2])
    a[I] = C * (-kappa * (d_1 * b[Im1_1] + d_2 * b[Im1_2 + b[Im2_2])
                          - (2 * (i + 1) - 1)
                          * (d_1 * a[Im1_1] + d_2 * a[Im1_2])
                          - (1 + i - 1) * a[Im2_2])




