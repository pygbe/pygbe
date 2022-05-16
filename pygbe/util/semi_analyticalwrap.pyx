STUFF = "Hi"
import cython
import numpy as np
cimport numpy as np

cdef extern from "semi_analyticalwrap.h":

	ctypedef double REAL

	void SA_wrap_arr_cy(REAL *y, int ySize, 
		REAL *x, int xSize, 
            	REAL *phi_Y, int pYSize, 
		REAL *dphi_Y, int dpYSize, 
            	REAL *phi_L, int pLSize, 
		REAL *dphi_L, int dpLSize,
            	REAL kappa, 
		int *same, int sameSize, 
		REAL *xk, int xkSize, 
		REAL *wk, int wkSize)

	void P2P_c_cy(REAL *MY, int MYSize, 
		REAL *dMY, int dMYSize, 
		REAL *ML, int MLSize, 
		REAL *dML, int dMLSize,    
        	REAL *triangle, int triangleSize, 
		int *tri, int triSize, 
		int *k, int kSize, 
        	REAL *xi, int xiSize, 
		REAL *yi, int yiSize, 
		REAL *zi, int ziSize,
        	REAL *s_xj, int s_xjSize, 
		REAL *s_yj, int s_yjSize, 
		REAL *s_zj, int s_zjSize,
        	REAL *xt, int xtSize, 
		REAL *yt, int ytSize, 
		REAL *zt, int ztSize,
        	REAL *m, int mSize, 
		REAL *mx, int mxSize, 
		REAL *my, int mySize, 
		REAL *mz, int mzSize, 
		REAL *mclean, int mcleanSize, 
		int *target, int targetSize,
        	REAL *Area, int AreaSize, 
		REAL *xk, int xkSize, 
        	REAL *wk, int wkSize, 
		REAL kappa, 
		REAL threshold, 
		REAL eps, 
		REAL w0, 
		REAL *aux, int auxSize)


def SA_wrap_arr(np.ndarray[REAL, ndim = 1] y, 
                   np.ndarray[REAL, ndim = 1] x,
                   np.ndarray[REAL, ndim = 1, mode = "c"] phi_Y, 
		   np.ndarray[REAL, ndim = 1, mode = "c"] dphi_Y, 
                   np.ndarray[REAL, ndim = 1, mode = "c"] phi_L, 
                   np.ndarray[REAL, ndim = 1, mode = "c"] dphi_L, 
		   REAL kappa,
                   np.ndarray[int, ndim = 1] same, 
                   np.ndarray[REAL, ndim = 1] xk,
                   np.ndarray[REAL, ndim = 1] wk):
	cdef np.int32_t ySize = len(y)
	cdef np.int32_t xSize = len(x)
	cdef np.int32_t pYSize = len(phi_Y)
	cdef np.int32_t dpYSize = len(dphi_Y)
	cdef np.int32_t pLSize = len(phi_L)
	cdef np.int32_t dpLSize = len(dphi_L)
	cdef np.int32_t sameSize = len(same)
	cdef np.int32_t xkSize = len(xk)
	cdef np.int32_t wkSize = len(wk)
	SA_wrap_arr_cy(<REAL*> &y[0] , <int> ySize, 
                   <REAL*> &x[0] , <int> xSize,
                   <REAL*> &phi_Y[0] , <int> pYSize, 
                   <REAL*> &dphi_Y[0], <int> dpYSize,
		   <REAL*> &phi_L[0] , <int> pLSize, 
                   <REAL*> &dphi_L[0], <int> dpLSize, 
		   <REAL> kappa,
		   <int*> &same[0], <int> sameSize,
                   <REAL*> &xk[0], <int> xkSize, 
                   <REAL*> &wk[0], <int> wkSize)

def P2P_c(np.ndarray[REAL, ndim = 1, mode = "c"] MY, 
                   np.ndarray[REAL, ndim = 1, mode = "c"] dMY,
                   np.ndarray[REAL, ndim = 1, mode = "c"] ML, 
		   np.ndarray[REAL, ndim = 1, mode = "c"] dML, 
                   np.ndarray[REAL, ndim = 1] triangle, 
                   np.ndarray[int, ndim = 1] tri,
                   np.ndarray[int, ndim = 1] k, 
                   np.ndarray[REAL, ndim = 1] xi,
                   np.ndarray[REAL, ndim = 1] yi,
		   np.ndarray[REAL, ndim = 1] zi,
		   np.ndarray[REAL, ndim = 1] s_xj,
		   np.ndarray[REAL, ndim = 1] s_yj,
		   np.ndarray[REAL, ndim = 1] s_zj,
		   np.ndarray[REAL, ndim = 1] xt,
		   np.ndarray[REAL, ndim = 1] yt,
		   np.ndarray[REAL, ndim = 1] zt,
		   np.ndarray[REAL, ndim = 1] m,
		   np.ndarray[REAL, ndim = 1] mx,
		   np.ndarray[REAL, ndim = 1] my,
		   np.ndarray[REAL, ndim = 1] mz,
		   np.ndarray[REAL, ndim = 1] mclean,
		   np.ndarray[int, ndim = 1] target,
		   np.ndarray[REAL, ndim = 1] Area,
		   np.ndarray[REAL, ndim = 1] xk,
		   np.ndarray[REAL, ndim = 1] wk,
		   REAL kappa,
		   REAL threshold, 
		   REAL eps, 
		   REAL w0,
		   np.ndarray[REAL, ndim = 1] aux):
	cdef np.int32_t MYSize = len(MY)
	cdef np.int32_t dMYSize = len(dMY)
	cdef np.int32_t MLSize = len(ML)
	cdef np.int32_t dMLSize = len(dML)
	cdef np.int32_t triangleSize = len(triangle)
	cdef np.int32_t triSize = len(tri) 
	cdef np.int32_t kSize = len(k)
	cdef np.int32_t xiSize = len(xi)
	cdef np.int32_t yiSize = len(yi)
	cdef np.int32_t ziSize = len(zi)
	cdef np.int32_t s_xjSize = len(s_xj)
	cdef np.int32_t s_yjSize = len(s_yj)
	cdef np.int32_t s_zjSize = len(s_zj)
	cdef np.int32_t xtSize = len(xt)
	cdef np.int32_t ytSize = len(yt)
	cdef np.int32_t ztSize = len(zt)
	cdef np.int32_t mSize = len(m)
	cdef np.int32_t mxSize = len(mx)
	cdef np.int32_t mySize = len(my)
	cdef np.int32_t mzSize = len(mz)
	cdef np.int32_t mcleanSize = len(mclean)
	cdef np.int32_t targetSize = len(target)
	cdef np.int32_t AreaSize = len(Area)
	cdef np.int32_t xkSize = len(xk)
	cdef np.int32_t wkSize = len(wk)
	cdef np.int32_t auxSize = len(aux)
	P2P_c_cy(<REAL*> &MY[0] , <int> MYSize, 
                   <REAL*> &dMY[0], <int> dMYSize,
                   <REAL*> &ML[0] , <int> MLSize, 
                   <REAL*> &dML[0], <int> dMLSize,
		   <REAL*> &triangle[0], <int> triangleSize, 
                   <int*> &tri[0], <int> triSize, 
		   <int*> &k[0], <int> kSize,
                   <REAL*> &xi[0], <int> xiSize, 
                   <REAL*> &yi[0], <int> yiSize, 
                   <REAL*> &zi[0], <int> ziSize, 
                   <REAL*> &s_xj[0], <int> s_xjSize, 
                   <REAL*> &s_yj[0], <int> s_yjSize, 
                   <REAL*> &s_zj[0], <int> s_zjSize, 
                   <REAL*> &xt[0], <int> xtSize, 
                   <REAL*> &yt[0], <int> ytSize, 
                   <REAL*> &zt[0], <int> ztSize, 
                   <REAL*> &m[0], <int> mSize, 
                   <REAL*> &mx[0], <int> mxSize, 
                   <REAL*> &my[0], <int> mySize, 
                   <REAL*> &mz[0], <int> mzSize, 
                   <REAL*> &mclean[0], <int> mcleanSize, 
                   <int*> &target[0], <int> targetSize, 
                   <REAL*> &Area[0], <int> AreaSize, 
                   <REAL*> &xk[0], <int> xkSize, 
                   <REAL*> &wk[0], <int> wkSize,
		   <REAL> kappa,
		   <REAL> threshold,
		   <REAL> eps,
		   <REAL> w0, 
                   <REAL*> &aux[0], <int> auxSize)
