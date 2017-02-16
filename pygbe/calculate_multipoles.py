import numpy
from numba import njit

@njit
def P2M(M, Md, x, y, z, m, mx, my, mz, xc, yc, zc, I, J, K):

    for i in range(len(I)):
        for j in range(len(x)):
            dx = xc - x[j]
            dy = yc - y[j]
            dz = zc - z[j]

            constant = dx**I[i] * dy**J[i] * dz**K[i]

            M[i] += m[j] * constant

            Md[i] -= mx[j] * I[i] * constant / dx;
            Md[i] -= my[j] * J[i] * constant / dy;
            Md[i] -= mz[j] * K[i] * constant / dz;

@njit
def M2M(MP, MC, dx, dy, dz, I, J, K, cI, cJ, cK, Imi, Jmj, Kmk, index, ptr):

    for i in range(len(MP)):
        ptr_start = ptr[i]
        size = ptr[i + 1] - ptr_start

        for j in range(size):
            mpt = ptr_start + j
            MP[i] += (MC[index[mpt]]
                      * cI[mpt] * cJ[mpt] * cK[mpt]
                      * dx**Imi[mpt] * dy**Jmj[mpt] * dz**Kmk[mpt])
