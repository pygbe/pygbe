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

    for CI in range(len(offTar)):
        CI_begin = offTar[CI]
        CI_end = offTar[CI] + sizeTar[CI]
        CJ_begin = offMlt[CI]
        CJ_end = offMlt[CI + 1]

        for CJ in range(CJ_begin, CJ_end):
            for i in range(CI_begin, CI_end):
                for ii in range(Nm):
                    a[ii] = 0

                dx = xi[i] - xc[CJ]
                dy = yi[i] - yc[CJ]
                dz = zi[i] - zc[CJ]

                getCoeff(a, dx, dy, dz, index, Nm, P, kappa, LorY)

                for j in range(Nm):
                    V_aux[i] += a[j] * M[CJ * Nm + j]
                    K_aux[i] += a[j] * Md[CJ * Nm + j]

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
    if yukawa or laplace:
        if yukawa:
            coef = yukawa_1
        else:
            coef = laplace_1

        for i in range(2, P):
            Cb = -kappa / (i + 1)
            C = 1 / ((1 + i) * R**2)

            I = index[getIndex(P, 1, i, 0)]
            Im1x = index[getIndex(P, 0, i, 0)]
            Im1y = I - (P + 2 - i - 1)
            Im2y = Im1y - (P + 2 - i)
            coef(I, a, b, i, Cb, C, dx, Im1x, dy, Im1y, Im2y, kappa)

            I = index[getIndex(P, 1, 0, i)]
            Im1x = index[getIndex(P, 0, 0, i)]
            Im1z = I - 1
            Im2z = I - 2
            coef(I, a, b, i, Cb, C, dx, Im1x, dz, Im1z, Im2z, kappa)

            I = index[getIndex(P, 0, 1, i)]
            Im1y = I - (P + 2 - 1)
            Im1z = I - 1
            Im2z = I - 2
            coef(I, a, b, i, Cb, C, dy, Im1y, dz, Im1z, Im2z, kappa)

            I = index[getIndex(P, i, 1, 0)]
            Im1y = I - (P + 2 - 1 - i)
            Im1x = index[getIndex(P, i - 1, 1, 0)]
            Im2x = index[getIndex(P, i - 2, 1, 0)]
            coef(I, a, b, i, Cb, C, dy, Im1y, dx, Im1x, Im2x, kappa)

            I = index[getIndex(P, i, 0, 1)]
            Im1z = I - 1
            Im1x = index[getIndex(P, i - 1, 0, 1)]
            Im2x = index[getIndex(P, i - 2, 0, 1)]
            coef(I, a, b, i, Cb, C, dz, Im1z, dx, Im1x, Im2x, kappa)

            I = index[getIndex(P, 0, i, 1)]
            Im1z = I - 1
            Im1y = I - (P + 2 - i)
            Im2y = Im1y - (P + 2 - i + 1)
            coef(I, a, b, i, Cb, C, dz, Im1z, dy, Im1y, Im2y, kappa)

    # porting from line 270 of multipole.cpp
    # Stay inside previous conditional? I think that's ok
        if yukawa:
            coef = yukawa_2
        else:
            coef = laplace_2

        for i in range(2, P + 1):
            for j in range(2, P + 1 - i):
                Cb = - kappa / (i + j)
                C = 1 / ((i + j) * R**2)

                I = index[getIndex(P, i, j, 0)]
                Im1x = index[getIndex(P, i - 1, j, 0)]
                Im2x = index[getIndex(P, i - 2, j, 0)]
                Im1y = I - (P + 2 - j - i)
                Im2y = Im1y - (P + 3 - j - i)
                coef(I, a, b, i, j, Cb, C, dx, Im1x, Im2x, dy, Im1y, Im2y, kappa)

                I = index[getIndex(P, i, 0, j)]
                Im1x = index[getIndex(P, i - 1, 0, j)]
                Im2x = index[getIndex(P, i - 2, 0, j)]
                Im1z = I - 1
                Im2z = I - 2
                coef(I, a, b, i, j, Cb, C, dx, Im1x, Im2x, dz, Im1z, Im2z, kappa)

                I = index[getIndex(P, 0, i, j)]
                Im1y = I - (P + 2 - i)
                Im2y = Im1y - (P + 3 - i)
                Im1z = I - 1
                Im2z = I - 2
                coef(I, a, b, i, j, Cb, C, dx, Im1x, Im2x, dz, Im1z, Im2z, kappa)

        if P > 2:
            Cb = - kappa / 3
            I = index[getIndex(P, 1, 1, 1)]
            Im1x = index[getIndex(P, 0, 1, 1)]
# BUG?      Im1y = index[getIndex(P, 1, 0, 1)]
            Im1y = I - P
            Im1z = I - 1
            if yukawa:
                b[I] = Cb * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z])
                a[I] = 1 / (3 * R**2) * (-kappa
                            * (dx * b[Im1x] + dy * b[Im1y] + dz * b[Im1z])
                            - 5 * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z]))
            else:
                a[I] = 1 / (3 * R**2) * (-5
                            * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z]))

                # pick up again at multipole.cpp line 345

            for i in range(2, P - 1):
                if yukawa:
                    coef = yukawa_3
                else:
                    coef = laplace_3
                Cb = -kappa / (2 + i)
                C = 1 / ((i + 2) * R**2)
                I = index[getIndex(P, i, 1, 1)]
                Im1x = index[getIndex(P, i - 1, 1, 1)]
                Im1y = I - (P + 2 - i - 1)
                Im1z = I - 1
                Im2x = index[getIndex(P, i - 2, 1, 1)]

                coef(I, a, b, i, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z, Im2x, kappa)

                I = index[getIndex(P, 1, i, 1)]
                Im1x = index[getIndex(P, 0, i, 1)]
                Im1y = I - (P + 2 - i - 1)
                Im2y = Im1y - (P + 3 - i - 1)
                Im1z = I - 1

                coef(I, a, b, i, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z, Im2y, kappa)

                I = index[getIndex(P, 1, 1, i)]
                Im1x = index[getIndex(P, 0, 1, i)]
                Im1y = I - P
                Im1z = I - 1
                Im2z = I - 2

                coef(I, a, b, i, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z, Im2z, kappa)

                # from multipole.cpp::397

        if P > 4:
            if yukawa:
                coef = yukawa_4
            else:
                coef = laplace_4
            for i in range(2, P - 2):
                for j in range(2, P - 1):
                    Cb = -kappa / (1 + i + j)
                    C = 1 / ((1 + i + j) * R**2)
                    C1 = -(2 * (1 + i + j) - 1)
                    C2 = i + j

                    I = index[getIndex(P, 1, i, j)]
                    Im1x = index[getIndex(P, 0, i, j)]
                    Im1y = I - (P + 2 - 1 - i)
                    Im2y = Im1y - (P + 3 - 1 - i)
                    Im1z = I - 1
                    Im2z = I - 2

                    coef(I, a, b, i, j, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z,
                         Im2y, Im2z, kappa)

                    I = index[getIndex(P, i, 1, j)]
                    Im1x = index[getIndex(P, i - 1, 1, j)]
                    Im1y = I - (P + 2 - i - 1)
                    Im2x = index[getIndex(P, i - 2, 1, j)]
                    Im1z = I - 1
                    Im2z = I - 2

                    coef(I, a, b, i, j, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z,
                         Im2x, Im2z, kappa)

                    I = index[getIndex(P, i, j, 1)]
                    Im1x = index[getIndex(P, i - 1, j, 1)]
                    Im2x = index[getIndex(P, i - 2, j, 1)]
                    Im1y = I - (P + 2 - i - j)
                    Im2y = Im1y - (P + 3 - i - j)
                    Im1z = I - 1

                    coef(I, a, b, i, j, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z,
                         Im2x, Im2y, kappa)

        # continue from mulipole.cpp::459
        if P > 5:
            for i in range(2, P - 3):
                for j in range(2, P - 1 - i):
                    for k in range(2, P + 1 - i - j):
                        Cb = -kappa / (i + j + k)
                        C = 1 / ((i + j + k) * R**2)
                        C1 = -(2 * (i + j + k) - 1)
                        C2 = i + j + k - 1

                        I = index[getIndex(P, i, j, k)]
                        Im1x = index[getIndex(P, i - 1, j, k)]
                        Im2x = index[getIndex(P, i - 2, j, k)]
                        Im1y = I - (P + 2 - i - j)
                        Im2y = Im1y - (P + 3 - i - j)
                        Im1z = I - 1
                        Im2z = I - 2

                        if yukawa:
                            b[I] = Cb * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z]
                                         + a[Im2x] + a[Im2y] + a[Im2z])
                            a[I] = C * (-kappa * (dx * b[Im1x] + dy * b[Im1y] +
                                                  dz * b[Im1z] + b[Im2x] + b[Im2y] +
                                                  b[Im2z])
                                        + C1 * (dx * a[Im1z]
                                                + dy * a[Im1y]
                                                + dz * a[Im1z])
                                        - C2 * (a[Im2x] + a[Im2y] + a[Im2z]))

@njit
def yukawa_1(I, a, b, i, Cb, C, d_1, Im1_1, d_2, Im1_2, Im2_2, kappa):
    b[I] = Cb * (d_1 * a[Im1_1] + d_2 * a[Im1_2] + a[Im2_2])
    a[I] = C * (-kappa * (d_1 * b[Im1_1] + d_2 * b[Im1_2] + b[Im2_2])
                          - (2 * (i + 1) - 1)
                          * (d_1 * a[Im1_1] + d_2 * a[Im1_2])
                          - (1 + i - 1) * a[Im2_2])

@njit
def yukawa_2(I, a, b, i, j, Cb, C, d_1, Im1_1, Im2_1, d_2, Im1_2, Im2_2, kappa):
    b[I] = Cb * (d_1 * a[Im1_1] + d_2 * a[Im1_2] + a[Im2_1] + a[Im2_2])
    a[I] = C * (-kappa * (d_1 * b[Im1_1] + d_2 * b[Im1_2] + b[Im2_1] + b[Im2_2])
                          - (2 * (i + 1) - 1)
                          * (d_1 * a[Im1_1] + d_2 * a[Im1_2])
                          - (i + j - 1) * (a[Im2_1] + a[Im2_2]))

@njit
def yukawa_3(I, a, b, i, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z, Im2, kappa):
    b[I] = Cb * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1x] + a[Im2])
    a[I] = C * (-kappa * (dx * b[Im1x] + dy*b[Im1y] + dz*b[Im1z] + b[Im2])
                -(2 * (i + 2) - 1)
                * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z]) - (i + 1)*a[Im2])

@njit
def yukawa_4(I, a, b, i, j, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z, Im2_1, Im2_2, kappa):
    C1 = -(2 * (1 + i + j) - 1)
    C2 = i + j
    b[I] = Cb * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z] + a[Im2_1] + a[Im2_2])
    a[I] = C * (-kappa * (dx * b[Im1x] + dy * b[Im1y] + dz * b[Im1z] + b[Im2_1] + b[Im2_2])
                + C1 * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z])
                - C2 * (a[Im2_1] + a[Im2_2]))



@njit
def laplace_1(I, a, b, i, Cb, C, d_1, Im1_1, d_2, Im1_2, Im2_2, kappa):
    a[I] = C * ( -(2 * (1 + i) - 1)
                 * (d_1 * a[Im1_1] + d_2 * a[Im1_2])
                 - (1 + i - 1) * a[Im2_2])

@njit
def laplace_2(I, a, b, i, j, Cb, C, d_1, Im1_1, Im2_1, d_2, Im1_2, Im2_2, kappa):
    a[I] = C * ( -(2 * (i + j) - 1)
                 * (d_1 * a[Im1_1] + d_2 * a[Im1_2])
                 - (i + j - 1) * (a[Im2_1] + a[Im2_2]))

@njit
def laplace_3(I, a, b, i, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z, Im2, kappa):
    a[I] = C * (-(2 * (i + 2) - 1) * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z])
                - (i + 1) * a[Im2])

@njit
def laplace_4(I, a, b, i, j, Cb, C, dx, dy, dz, Im1x, Im1y, Im1z, Im2_1, Im2_2, kappa):
    C1 = -(2 * (1 + i + j) - 1)
    C2 = i + j
    a[I] = C * (C1 * (dx * a[Im1x] + dy * a[Im1y] + dz * a[Im1z])
                - C2 * (a[Im2_1] + a[Im2_2]))
