STUFF = "Hi"
import cython
import numpy as np
cimport numpy as np

cdef extern from "multipole.h":

	ctypedef double REAL

	void multipole_c_cy(REAL *K_aux , int K_auxSize, 
                 REAL *V_aux , int V_auxSize,
                   REAL *M , int MSize, 
                   REAL *Md, int MdSize, 
                   REAL *dx, int dxSize, 
                   REAL *dy, int dySize, 
                   REAL *dz, int dzSize,
                   int *index, int indexSize,
                   int P, REAL kappa, int Nm, int LorY)

	void multipole_sort_cy(REAL *K_aux , int K_auxSize, 
                    REAL *V_aux , int V_auxSize,
                    int *offTar, int offTarSize,
                    int *sizeTar, int sizeTarSize,
                    int *offMlt, int offMltSize,
                    REAL *M , int MSize, 
                    REAL *Md, int MdSize, 
                    REAL *xi, int xiSize, 
                    REAL *yi, int yiSize, 
                    REAL *zi, int ziSize,
                    REAL *xisort, int xisortSize, 
                    REAL *yisort, int yisortSize, 
                    REAL *zisort, int zisortSize,
                    int *unsort, int unsortSize,
                    int *sort, int sort,
                    REAL *xc, int xcSize, 
                    REAL *yc, int ycSize, 
                    REAL *zc, int zcSize,
                    int *index, int indexSize,
                    int P, REAL kappa, int Nm, int LorY)

	void multipoleKt_sort_cy(REAL *Ktx_aux , int Ktx_auxSize, 
                    REAL *Kty_aux , int Kty_auxSize,
                    REAL *Ktz_aux , int Ktz_auxSize,
                    int *offTar, int offTarSize,
                    int *sizeTar, int sizeTarSize,
                    int *offMlt, int offMltSize,
                    REAL *M , int MSize, 
                    REAL *xi, int xiSize, 
                    REAL *yi, int yiSize, 
                    REAL *zi, int ziSize,
                    REAL *xc, int xcSize, 
                    REAL *yc, int ycSize, 
                    REAL *zc, int zcSize,
                    int *index, int indexSize,
                    int P, REAL kappa, int Nm, int LorY)

	void getIndex_arr_cy(int P, int N, int *indices, int indicesSize, int *ii, int iiSize, int *jj, int jjSize, int *kk, int kkSize)

	int setIndex_cy(int P, int i, int j, int k)




def multipole_c(np.ndarray[REAL, ndim = 1, mode = "c"] K_aux, 
                   np.ndarray[REAL, ndim = 1, mode = "c"] V_aux,
                   np.ndarray[REAL, ndim = 1] M, 
                   np.ndarray[REAL, ndim = 1] Md, 
                   np.ndarray[REAL, ndim = 1] dx, 
                   np.ndarray[REAL, ndim = 1] dy, 
                   np.ndarray[REAL, ndim = 1] dz,
                   np.ndarray[int, ndim = 1] index,
                   int P, REAL kappa, int Nm, int LorY):
	cdef np.int32_t K_auxSize = len(K_aux)
	cdef np.int32_t V_auxSize = len(V_aux)
	cdef np.int32_t MSize = len(M)
	cdef np.int32_t MdSize = len(Md)
	cdef np.int32_t dxSize = len(dx)
	cdef np.int32_t dySize = len(dy)
	cdef np.int32_t dzSize = len(dz)
	cdef np.int32_t indexSize = len(index)
	multipole_c_cy(<REAL*> &K_aux[0] , <int> K_auxSize, 
                   <REAL*> &V_aux[0] , <int> V_auxSize,
                   <REAL*> &M[0] , <int> MSize, 
                   <REAL*> &Md[0], <int> MdSize, 
                   <REAL*> &dx[0], <int> dxSize, 
                   <REAL*> &dy[0], <int> dySize, 
                   <REAL*> &dz[0], <int> dzSize,
                   <int*> &index[0], <int> indexSize,
                   <int> P, <REAL> kappa, <int> Nm, <int> LorY)

def multipole_sort(np.ndarray[REAL, ndim = 1, mode = "c"] K_aux, 
                    np.ndarray[REAL, ndim = 1, mode = "c"] V_aux,
                    np.ndarray[int, ndim = 1] offTar,
                    np.ndarray[int, ndim = 1] sizeTar,
                    np.ndarray[int, ndim = 1] offMlt,
                    np.ndarray[REAL, ndim = 1] M, 
                    np.ndarray[REAL, ndim = 1] Md, 
                    np.ndarray[REAL, ndim = 1] xi, 
                    np.ndarray[REAL, ndim = 1] yi, 
                    np.ndarray[REAL, ndim = 1] zi,
                    np.ndarray[REAL, ndim = 1] xisort, 
                    np.ndarray[REAL, ndim = 1] yisort, 
                    np.ndarray[REAL, ndim = 1] zisort,
                    np.ndarray[int, ndim = 1] unsort,
                    np.ndarray[int, ndim = 1] sort,
                    np.ndarray[REAL, ndim = 1] xc, 
                    np.ndarray[REAL, ndim = 1] yc, 
                    np.ndarray[REAL, ndim = 1] zc,
                    np.ndarray[int, ndim = 1] index,
                    int P, REAL kappa, int Nm, int LorY):
	cdef np.int32_t K_auxSize = len(K_aux)
	cdef np.int32_t V_auxSize = len(V_aux)
	cdef np.int32_t offTarSize = len(offTar)
	cdef np.int32_t sizeTarSize = len(sizeTar)
	cdef np.int32_t offMltSize = len(offMlt)
	cdef np.int32_t MSize = len(M)
	cdef np.int32_t MdSize = len(Md)
	cdef np.int32_t xiSize = len(xi)
	cdef np.int32_t yiSize = len(yi)
	cdef np.int32_t ziSize = len(zi)
	cdef np.int32_t xisortSize = len(xisort)
	cdef np.int32_t yisortSize = len(yisort)
	cdef np.int32_t zisortSize = len(zisort)
	cdef np.int32_t unsortSize = len(unsort)
	cdef np.int32_t sortSize = len(sort)
	cdef np.int32_t xcSize = len(xc)
	cdef np.int32_t ycSize = len(yc)
	cdef np.int32_t zcSize = len(zc)
	cdef np.int32_t indexSize = len(index)
	multipole_sort_cy(<REAL*> &K_aux[0] , <int> K_auxSize, 
                    <REAL*> &V_aux[0] , <int> V_auxSize,
                    <int*> &offTar[0], <int> offTarSize,
                    <int*> &sizeTar[0], <int> sizeTarSize,
                    <int*> &offMlt[0], <int> offMltSize,
                    <REAL*> &M[0] , <int> MSize, 
                    <REAL*> &Md[0], <int> MdSize, 
                    <REAL*> &xi[0], <int> xiSize, 
                    <REAL*> &yi[0], <int> yiSize, 
                    <REAL*> &zi[0], <int> ziSize,
                    <REAL*> &xisort[0], <int> xisortSize, 
                    <REAL*> &yisort[0], <int> yisortSize, 
                    <REAL*> &zisort[0], <int> zisortSize,
                    <int*> &unsort[0], <int> unsortSize,
                    <int*> &sort[0], <int> sortSize,
                    <REAL*> &xc[0], <int> xcSize, 
                    <REAL*> &yc[0], <int> ycSize, 
                    <REAL*> &zc[0], <int> zcSize,
                    <int*> &index[0], <int> indexSize,
                    <int> P, <REAL> kappa, <int> Nm, <int> LorY)


def multipoleKt_sort(np.ndarray[REAL, ndim = 1, mode = "c"] Ktx_aux, 
                    np.ndarray[REAL, ndim = 1, mode = "c"] Kty_aux,
                    np.ndarray[REAL, ndim = 1, mode = "c"] Ktz_aux,
                    np.ndarray[int, ndim = 1] offTar,
                    np.ndarray[int, ndim = 1] sizeTar,
                    np.ndarray[int, ndim = 1] offMlt,
                    np.ndarray[REAL, ndim = 1] M, 
                    np.ndarray[REAL, ndim = 1] xi, 
                    np.ndarray[REAL, ndim = 1] yi, 
                    np.ndarray[REAL, ndim = 1] zi,
                    np.ndarray[REAL, ndim = 1] xc, 
                    np.ndarray[REAL, ndim = 1] yc, 
                    np.ndarray[REAL, ndim = 1] zc,
                    np.ndarray[int, ndim = 1] index,
                    int P, REAL kappa, int Nm, int LorY):
	cdef np.int32_t Ktx_auxSize = len(Ktx_aux)
	cdef np.int32_t Kty_auxSize = len(Kty_aux)
	cdef np.int32_t Ktz_auxSize = len(Ktz_aux)
	cdef np.int32_t offTarSize = len(offTar)
	cdef np.int32_t sizeTarSize = len(sizeTar)
	cdef np.int32_t offMltSize = len(offMlt)
	cdef np.int32_t MSize = len(M)
	cdef np.int32_t xiSize = len(xi)
	cdef np.int32_t yiSize = len(yi)
	cdef np.int32_t ziSize = len(zi)
	cdef np.int32_t xcSize = len(xc)
	cdef np.int32_t ycSize = len(yc)
	cdef np.int32_t zcSize = len(zc)
	cdef np.int32_t indexSize = len(index)
	multipoleKt_sort_cy(<REAL*> &Ktx_aux[0] , <int> Ktx_auxSize, 
                    <REAL*> &Kty_aux[0] , <int> Kty_auxSize,
                    <REAL*> &Ktz_aux[0] , <int> Ktz_auxSize,
                    <int*> &offTar[0], <int> offTarSize,
                    <int*> &sizeTar[0], <int> sizeTarSize,
                    <int*> &offMlt[0], <int> offMltSize,
                    <REAL*> &M[0], <int> MSize, 
                    <REAL*> &xi[0], <int> xiSize, 
                    <REAL*> &yi[0], <int> yiSize, 
                    <REAL*> &zi[0], <int> ziSize,
                    <REAL*> &xc[0], <int> xcSize, 
                    <REAL*> &yc[0], <int> ycSize, 
                    <REAL*> &zc[0], <int> zcSize,
                    <int*> &index[0], <int> indexSize,
                    <int> P, <REAL> kappa, <int> Nm, <int> LorY)

def getIndex_arr(int P, int N, np.ndarray[int, ndim = 1, mode = "c"] indices, np.ndarray[int, ndim = 1] ii, np.ndarray[int, ndim = 1] jj, np.ndarray[int, ndim = 1] kk):
	cdef np.int32_t indicesSize = len(indices)
	cdef np.int32_t iiSize = len(ii)
	cdef np.int32_t jjSize = len(jj)
	cdef np.int32_t kkSize = len(kk)
	getIndex_arr_cy(<int> P, <int> N, <int*> &indices[0], <int> indicesSize, <int*> &ii[0], <int> iiSize, <int*> &jj[0], <int> jjSize, <int*> &kk[0], <int> kkSize)


def setIndex(int P, int i, int j, int k):
	return setIndex_cy(<int> P, <int> i, <int> j, <int> k)

