STUFF = "Hi"
import cython
import numpy as np
cimport numpy as np

cdef extern from "direct.h":

	ctypedef double REAL

	void computeDiagonal_cy(REAL *VL, int VLSize, REAL *KL, int KLSize, REAL *VY, int VYSize, REAL *KY, int KYSize, 
                    REAL *triangle, int triangleSize, REAL *centers, int centersSize, REAL kappa,
                    REAL K_diag, REAL V_diag, REAL *xk, int xkSize, REAL *wk, int wkSize)

	void direct_sort_cy(REAL *K_aux, int K_auxSize, REAL *V_aux, int V_auxSize, int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
        	int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
        	REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
        	REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        	REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
        	int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, int *offSrc, int offSrcSize, int *offTwg, int offTwgSize,  
        	int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
        	REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize,
        	REAL kappa, REAL threshold, REAL eps, REAL w0, REAL *aux, int auxSize)

	void directKt_sort_cy(REAL *Ktx_aux, int Ktx_auxSize, REAL *Kty_aux, int Kty_auxSize, REAL *Ktz_aux, int Ktz_auxSize, 
        	int LorY, REAL *triangle, int triangleSize,
        	int *k, int kSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, REAL *s_zj, int s_zjSize, 
        	REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        	REAL *m, int mSize, REAL *mKclean, int mKcleanSize,
        	int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, 
        	int *offSrc, int offSrcSize, int *offTwg, int offTwgSize, REAL *Area, int AreaSize,
        	REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, REAL kappa, REAL threshold, REAL eps, REAL *aux, int auxSize)

	void direct_c_cy(int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
        	int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
        	REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
        	REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        	REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
        	int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
        	REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, 
        	REAL kappa, REAL threshold, REAL eps, REAL w0, int AI_int, REAL *phi_reac, int phi_reacSize)

	void coulomb_direct_cy(REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize, 
                    REAL *m, int mSize, REAL *K_aux, int K_auxSize);
        	


def computeDiagonal(np.ndarray[REAL, ndim = 1, mode = "c"] VL, np.ndarray[REAL, ndim = 1, mode = "c"] KL, np.ndarray[REAL, ndim = 1, mode = "c"] VY, np.ndarray[REAL, ndim = 1, mode = "c"] KY, np.ndarray[REAL, ndim = 1, mode = "c"] triangle, np.ndarray[REAL, ndim = 1, mode = "c"] centers, REAL kappa, REAL K_diag, REAL V_diag, np.ndarray[REAL, ndim = 1, mode = "c"] xk, np.ndarray[REAL, ndim = 1, mode = "c"] wk):
	cdef np.int32_t VLSize = len(VL)
	cdef np.int32_t KLSize = len(KL)
	cdef np.int32_t VYSize = len(VY)
	cdef np.int32_t KYSize = len(KY)
	cdef np.int32_t triangleSize = len(triangle)
	cdef np.int32_t centersSize = len(centers)
	cdef np.int32_t xkSize = len(xk)
	cdef np.int32_t wkSize = len(wk)
	computeDiagonal_cy(<REAL*> &VL[0], <int> VLSize, <REAL*> &KL[0], <int> KLSize, <REAL*> &VY[0], <int> VYSize, <REAL*> &KY[0], <int> KYSize, <REAL*> &triangle[0], <int> triangleSize, <REAL*> &centers[0], <int> centersSize, <REAL> kappa, <REAL> K_diag, <REAL> V_diag, <REAL*> &xk[0], <int> xkSize, <REAL*> &wk[0], <int> wkSize)


def direct_sort(np.ndarray[REAL, ndim = 1, mode = "c"] K_aux, np.ndarray[REAL, ndim = 1, mode = "c"] V_aux, int LorY, REAL K_diag, REAL V_diag, int IorE, np.ndarray[REAL, ndim = 1 ] triangle, np.ndarray[int, ndim = 1 ] tri, np.ndarray[int, ndim = 1 ] k, np.ndarray[REAL, ndim = 1 ] xi, np.ndarray[REAL, ndim = 1 ] yi, np.ndarray[REAL, ndim = 1 ] zi, np.ndarray[REAL, ndim = 1 ] s_xj, np.ndarray[REAL, ndim = 1 ] s_yj, np.ndarray[REAL, ndim = 1 ] s_zj, np.ndarray[REAL, ndim = 1 ] xt, np.ndarray[REAL, ndim = 1 ] yt, np.ndarray[REAL, ndim = 1 ] zt, np.ndarray[REAL, ndim = 1 ] m, np.ndarray[REAL, ndim = 1 ] mx, np.ndarray[REAL, ndim = 1 ] my, np.ndarray[REAL, ndim = 1 ] mz, np.ndarray[REAL, ndim = 1 ] mKclean, np.ndarray[REAL, ndim = 1 ] mVclean, np.ndarray[int, ndim = 1 ] interList, np.ndarray[int, ndim = 1 ] offTar, np.ndarray[int, ndim = 1 ] sizeTar, np.ndarray[int, ndim = 1 ] offSrc, np.ndarray[int, ndim = 1 ] offTwg, np.ndarray[int, ndim = 1 ] target, np.ndarray[REAL, ndim = 1 ] Area, np.ndarray[REAL, ndim = 1 ] sglInt_int, np.ndarray[REAL, ndim = 1 ] sglInt_ext, np.ndarray[REAL, ndim = 1 ] xk, np.ndarray[REAL, ndim = 1 ] wk, np.ndarray[REAL, ndim = 1 ] Xsk, np.ndarray[REAL, ndim = 1 ] Wsk, REAL kappa, REAL threshold, REAL eps, REAL w0, np.ndarray[REAL, ndim = 1 ] aux):
	cdef np.int32_t K_auxSize = len(K_aux)
	cdef np.int32_t V_auxSize = len(V_aux)
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
	cdef np.int32_t mKcleanSize = len(mKclean)
	cdef np.int32_t mVcleanSize = len(mVclean)
	cdef np.int32_t interListSize = len(interList)
	cdef np.int32_t offTarSize = len(offTar)
	cdef np.int32_t sizeTarSize = len(sizeTar)
	cdef np.int32_t offSrcSize = len(offSrc)
	cdef np.int32_t offTwgSize = len(offTwg)
	cdef np.int32_t targetSize = len(target)
	cdef np.int32_t AreaSize = len(Area)
	cdef np.int32_t sglInt_intSize = len(sglInt_int)
	cdef np.int32_t sglInt_extSize = len(sglInt_ext)
	cdef np.int32_t xkSize = len(xk)
	cdef np.int32_t wkSize = len(wk)
	cdef np.int32_t XskSize = len(Xsk)
	cdef np.int32_t WskSize = len(Wsk)
	cdef np.int32_t auxSize = len(aux)
	direct_sort_cy(<REAL*> &K_aux[0], <int> K_auxSize, <REAL*> &V_aux[0], <int> V_auxSize, <int> LorY, <REAL> K_diag, <REAL> V_diag, <int> IorE, <REAL*> &triangle[0], <int> triangleSize,
        	<int*> &tri[0], <int> triSize, <int*> &k[0], <int> kSize, <REAL*> &xi[0], <int> xiSize, <REAL*> &yi[0], <int> yiSize, 
        	<REAL*> &zi[0], <int> ziSize, <REAL*> &s_xj[0], <int> s_xjSize, <REAL*> &s_yj[0], <int> s_yjSize, 
        	<REAL*> &s_zj[0], <int> s_zjSize, <REAL*> &xt[0], <int> xtSize, <REAL*> &yt[0], <int> ytSize, <REAL*> &zt[0], <int> ztSize,
        	<REAL*> &m[0], <int> mSize, <REAL*> &mx[0], <int> mxSize, <REAL*> &my[0], <int> mySize, <REAL*> &mz[0], <int> mzSize, <REAL*> &mKclean[0], <int> mKcleanSize, <REAL*> &mVclean[0], <int> mVcleanSize,
        	<int*> &interList[0], <int> interListSize, <int*> &offTar[0], <int> offTarSize, <int*> &sizeTar[0], <int> sizeTarSize, <int*> &offSrc[0], <int> offSrcSize, <int*> &offTwg[0], <int> offTwgSize,  
        	<int*> &target[0], <int> targetSize, <REAL*> &Area[0], <int> AreaSize, <REAL*> &sglInt_int[0], <int> sglInt_intSize, <REAL*> &sglInt_ext[0], <int> sglInt_extSize, 
        	<REAL*> &xk[0], <int> xkSize, <REAL*> &wk[0], <int> wkSize, <REAL*> &Xsk[0], <int> XskSize, <REAL*> &Wsk[0], <int> WskSize,
        	<REAL> kappa, <REAL> threshold, <REAL> eps, <REAL> w0, <REAL*> &aux[0], <int> auxSize)


def directKt_sort(np.ndarray[REAL, ndim = 1, mode = "c"] Ktx_aux, np.ndarray[REAL, ndim = 1, mode = "c"] Kty_aux, np.ndarray[REAL, ndim = 1, mode = "c"] Ktz_aux, 
        	int LorY, np.ndarray[REAL, ndim = 1 ] triangle,
        	np.ndarray[int, ndim = 1 ] k, np.ndarray[REAL, ndim = 1 ] s_xj, np.ndarray[REAL, ndim = 1 ] s_yj, np.ndarray[REAL, ndim = 1 ] s_zj, 
        	np.ndarray[REAL, ndim = 1 ] xt, np.ndarray[REAL, ndim = 1 ] yt, np.ndarray[REAL, ndim = 1 ] zt,
        	np.ndarray[REAL, ndim = 1 ] m, np.ndarray[REAL, ndim = 1 ] mKclean,
        	np.ndarray[int, ndim = 1 ] interList, np.ndarray[int, ndim = 1 ] offTar, np.ndarray[int, ndim = 1 ] sizeTar, 
        	np.ndarray[int, ndim = 1 ] offSrc, np.ndarray[int, ndim = 1 ] offTwg, np.ndarray[REAL, ndim = 1 ] Area,
        	np.ndarray[REAL, ndim = 1 ] Xsk, np.ndarray[REAL, ndim = 1 ] Wsk, REAL kappa, REAL threshold, REAL eps, np.ndarray[REAL, ndim = 1 ] aux):
	cdef np.int32_t Ktx_auxSize = len(Ktx_aux)
	cdef np.int32_t Kty_auxSize = len(Kty_aux)
	cdef np.int32_t Ktz_auxSize = len(Ktz_aux)
	cdef np.int32_t triangleSize = len(triangle)
	cdef np.int32_t kSize = len(k)
	cdef np.int32_t s_xjSize = len(s_xj)
	cdef np.int32_t s_yjSize = len(s_yj)
	cdef np.int32_t s_zjSize = len(s_zj)
	cdef np.int32_t xtSize = len(xt)
	cdef np.int32_t ytSize = len(yt)
	cdef np.int32_t ztSize = len(zt)
	cdef np.int32_t mSize = len(m)
	cdef np.int32_t mKcleanSize = len(mKclean)
	cdef np.int32_t interListSize = len(interList)
	cdef np.int32_t offTarSize = len(offTar)
	cdef np.int32_t sizeTarSize = len(sizeTar)
	cdef np.int32_t offSrcSize = len(offSrc)
	cdef np.int32_t offTwgSize = len(offTwg)
	cdef np.int32_t AreaSize = len(Area)
	cdef np.int32_t XskSize = len(Xsk)
	cdef np.int32_t WskSize = len(Wsk)
	cdef np.int32_t auxSize = len(aux)
	directKt_sort_cy(<REAL*> &Ktx_aux[0], <int> Ktx_auxSize, <REAL*> &Kty_aux[0], <int> Kty_auxSize, <REAL*> &Ktz_aux[0], <int> Ktz_auxSize, 
        	<int> LorY, <REAL*> &triangle[0], <int> triangleSize,
        	<int*> &k[0], <int> kSize, <REAL*> &s_xj[0], <int> s_xjSize, <REAL*> &s_yj[0], <int> s_yjSize, <REAL*> &s_zj[0], <int> s_zjSize, 
        	<REAL*> &xt[0], <int> xtSize, <REAL*> &yt[0], <int> ytSize, <REAL*> &zt[0], <int> ztSize,
        	<REAL*> &m[0], <int> mSize, <REAL*> &mKclean[0], <int> mKcleanSize,
        	<int*> &interList[0], <int> interListSize, <int*> &offTar[0], <int> offTarSize, <int*> &sizeTar[0], <int> sizeTarSize, 
        	<int*> &offSrc[0], <int> offSrcSize, <int*> &offTwg[0], <int> offTwgSize, <REAL*> &Area[0], <int> AreaSize,
        	<REAL*> &Xsk[0], <int> XskSize, <REAL*> &Wsk[0], <int> WskSize, <REAL> kappa, <REAL> threshold, <REAL> eps, <REAL*> &aux[0], <int> auxSize)


def direct_c(int LorY, REAL K_diag, REAL V_diag, int IorE, np.ndarray[REAL, ndim = 1 ] triangle, np.ndarray[int, ndim = 1 ] tri, np.ndarray[int, ndim = 1 ] k, np.ndarray[REAL, ndim = 1 ] xi, np.ndarray[REAL, ndim = 1 ] yi, np.ndarray[REAL, ndim = 1 ] zi, np.ndarray[REAL, ndim = 1 ] s_xj, np.ndarray[REAL, ndim = 1 ] s_yj, np.ndarray[REAL, ndim = 1 ] s_zj, np.ndarray[REAL, ndim = 1 ] xt, np.ndarray[REAL, ndim = 1 ] yt, np.ndarray[REAL, ndim = 1 ] zt, np.ndarray[REAL, ndim = 1 ] m, np.ndarray[REAL, ndim = 1 ] mx, np.ndarray[REAL, ndim = 1 ] my, np.ndarray[REAL, ndim = 1 ] mz, np.ndarray[REAL, ndim = 1 ] mKclean, np.ndarray[REAL, ndim = 1 ] mVclean, np.ndarray[int, ndim = 1 ] target, np.ndarray[REAL, ndim = 1 ] Area, np.ndarray[REAL, ndim = 1 ] sglInt_int, np.ndarray[REAL, ndim = 1 ] sglInt_ext, np.ndarray[REAL, ndim = 1 ] xk, np.ndarray[REAL, ndim = 1 ] wk, np.ndarray[REAL, ndim = 1 ] Xsk, np.ndarray[REAL, ndim = 1 ] Wsk, REAL kappa, REAL threshold, REAL eps, REAL w0, int AI_int, np.ndarray[REAL, ndim = 1 ] phi_reac):
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
	cdef np.int32_t mKcleanSize = len(mKclean)
	cdef np.int32_t mVcleanSize = len(mVclean)
	cdef np.int32_t targetSize = len(target)
	cdef np.int32_t AreaSize = len(Area)
	cdef np.int32_t sglInt_intSize = len(sglInt_int)
	cdef np.int32_t sglInt_extSize = len(sglInt_ext)
	cdef np.int32_t xkSize = len(xk)
	cdef np.int32_t wkSize = len(wk)
	cdef np.int32_t XskSize = len(Xsk)
	cdef np.int32_t WskSize = len(Wsk)
	cdef np.int32_t phi_reacSize = len(phi_reac)

	direct_c_cy(<int> LorY, <REAL> K_diag, <REAL> V_diag, <int> IorE, <REAL*> &triangle[0], <int> triangleSize,
        	<int*> &tri[0], <int> triSize, <int*> &k[0], <int> kSize, <REAL*> &xi[0], <int> xiSize, <REAL*> &yi[0], <int> yiSize, 
        	<REAL*> &zi[0], <int> ziSize, <REAL*> &s_xj[0], <int> s_xjSize, <REAL*> &s_yj[0], <int> s_yjSize, 
        	<REAL*> &s_zj[0], <int> s_zjSize, <REAL*> &xt[0], <int> xtSize, <REAL*> &yt[0], <int> ytSize, <REAL*> &zt[0], <int> ztSize,
        	<REAL*> &m[0], <int> mSize, <REAL*> &mx[0], <int> mxSize, <REAL*> &my[0], <int> mySize, <REAL*> &mz[0], <int> mzSize, <REAL*> &mKclean[0], <int> mKcleanSize, <REAL*> &mVclean[0], <int> mVcleanSize, 
        	<int*> &target[0], <int> targetSize, <REAL*> &Area[0], <int> AreaSize, <REAL*> &sglInt_int[0], <int> sglInt_intSize, <REAL*> &sglInt_ext[0], <int> sglInt_extSize, 
        	<REAL*> &xk[0], <int> xkSize, <REAL*> &wk[0], <int> wkSize, <REAL*> &Xsk[0], <int> XskSize, <REAL*> &Wsk[0], <int> WskSize,
        	<REAL> kappa, <REAL> threshold, <REAL> eps, <REAL> w0, <int> AI_int, <REAL*> &phi_reac[0], <int> phi_reacSize)


def coulomb_direct(np.ndarray[REAL, ndim = 1, mode = "c"] xt, np.ndarray[REAL, ndim = 1, mode = "c"] yt, np.ndarray[REAL, ndim = 1, mode = "c"] zt, 
                   np.ndarray[REAL, ndim = 1, mode = "c"] m, np.ndarray[REAL, ndim = 1, mode = "c"] K_aux):
	cdef np.int32_t xtSize = len(xt)
	cdef np.int32_t ytSize = len(yt)
	cdef np.int32_t ztSize = len(zt)
	cdef np.int32_t mSize = len(m)
	cdef np.int32_t K_auxSize = len(K_aux)
	coulomb_direct_cy(<REAL*> &xt[0], <int> xtSize, <REAL*> &yt[0], <int> ytSize, <REAL*> &zt[0], <int> ztSize, 
                    <REAL*> &m[0], <int> mSize, <REAL*> &K_aux[0], <int> K_auxSize)
