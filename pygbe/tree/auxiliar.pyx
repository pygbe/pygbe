STUFF = "Hi"
import cython
import numpy as np
cimport numpy as np

cdef extern from "auxiliar.h":

	ctypedef double REAL

	void calc_aux_cy(REAL *q , int qSize, 
                    REAL *xq0 , int xq0Size,
                    REAL *xq1 , int xq1Size,
                    REAL *xq2 , int xq2Size,
                    REAL *xi , int xiSize,
                    REAL *yi , int yiSize,
                    REAL *zi , int ziSize,
                    REAL *normal0 , int normal0Size, 
                    REAL *normal1 , int normal1Size, 
                    REAL *normal2 , int normal2Size,
                    int stype,
                    REAL *aux , int auxSize,
                    REAL E)


def calc_aux(np.ndarray[REAL, ndim = 1] q, 
                   np.ndarray[REAL, ndim = 1] xq0,
                   np.ndarray[REAL, ndim = 1] xq1,
                   np.ndarray[REAL, ndim = 1] xq2, 
                   np.ndarray[REAL, ndim = 1] xi, 
                   np.ndarray[REAL, ndim = 1] yi, 
                   np.ndarray[REAL, ndim = 1] zi, 
                   np.ndarray[REAL, ndim = 1] normal0,
                   np.ndarray[REAL, ndim = 1] normal1,
                   np.ndarray[REAL, ndim = 1] normal2,
                   int stype,
                   np.ndarray[REAL, ndim = 1, mode = "c"] aux,
                   REAL E):
	cdef np.int32_t qSize = len(q)
	cdef np.int32_t xq0Size = len(xq0)
	cdef np.int32_t xq1Size = len(xq1)
	cdef np.int32_t xq2Size = len(xq2)
	cdef np.int32_t xiSize = len(xi)
	cdef np.int32_t yiSize = len(yi)
	cdef np.int32_t ziSize = len(zi)
	cdef np.int32_t normal0Size = len(normal0)
	cdef np.int32_t normal1Size = len(normal1)
	cdef np.int32_t normal2Size = len(normal2)
	cdef np.int32_t auxSize = len(aux)
	calc_aux_cy(<REAL*> &q[0] , <int> qSize,
                   <REAL*> &xq0[0] , <int> xq0Size,
                   <REAL*> &xq1[0] , <int> xq1Size,
                   <REAL*> &xq2[0] , <int> xq2Size,
                   <REAL*> &xi[0] , <int> xiSize,
                   <REAL*> &yi[0] , <int> yiSize,
                   <REAL*> &zi[0] , <int> ziSize,
                   <REAL*> &normal0[0] , <int> normal0Size,
                   <REAL*> &normal1[0] , <int> normal1Size,
                   <REAL*> &normal2[0] , <int> normal2Size,
                   <int> stype,
                   <REAL*> &aux[0] , <int> auxSize,
                   <REAL> E)
