import numba
import numpy

@numba.njit(cache=True, nogil=True)
def P2M(M, Md, x, y, z, m, mx, my, mz, xc, yc, zc, I, J, K):
    for i in range(len(I)):
        for j in range(len(x)):
            dx = xc - x[j]
            dy = yc - y[j]
            dz = zc - z[j]
            dxI   = dx**I[i]
            dyJ   = dy**J[i]
            dzK   = dz**K[i]
            constant = dxI * dyJ * dzK
            M[i] += m[j] * constant
            Md[i] -= mx[j] * I[i] * constant / dx
            Md[i] -= my[j] * J[i] * constant / dy
            Md[i] -= mz[j] * K[i] * constant / dz

    return M, Md


@numba.njit(cache=True, nogil=True)
def M2M(MP, MC, dx, dy, dz, I, J, K, cI, cJ, cK, Imi, Jmj, Kmk, index, ptr):
    for i in range(len(MP)):
        ptr_start = ptr[i]
        size = ptr[i + 1] - ptr_start

        for j in range(size):
            Mptr = ptr_start + j
            MP[i] += MC[index[Mptr]] * cI[Mptr] * cJ[Mptr] * cK[Mptr] * dx**Imi[Mptr] * dy**Jmj[Mptr] * dz**Kmk[Mptr]


    return MP
