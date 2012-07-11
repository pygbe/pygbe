%module getCoeffwrap

%{
#define SWIG_FILE_WITH_INIT
extern void getCoeff(double *aY, int aYSize, 
                    double *axY, int axYSize, 
                    double *ayY, int ayYSize, 
                    double *azY, int azYSize, 
                    double *aL, int aLSize, 
                    double *axL, int axLSize, 
                    double *ayL, int ayLSize, 
                    double *azL, int azLSize, 
                    double dx, double dy, double dz, int P, double kappa);
extern void multipole(double *Lap, int LSize,
                      double *dL, int dLSize,
                      double *Y, int YSize,
                      double *dY, int dYSize,
                      double *M, int MSize,
                      double *Mx, int MxSize,
                      double *My, int MySize,
                      double *Mz, int MzSize,
                      double *dxa, int dxaSize,
                      double *dya, int dyaSize,
                      double *dza, int dzaSize,
                      int P, double kappa, int Nm, double E_Hat);
extern void getIndex_arr(int P, int N, 
                        int *indices, int indicesSize,
                        int * ii    , int iiSize,
                        int * jj    , int jjSize,
                        int * kk    , int kkSize);
extern int getIndex(int P, int i, int j, int k);
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1){(double *Lap, int LSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dL, int dLSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Y, int YSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dY, int dYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *aY, int aYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *axY, int axYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *ayY, int ayYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *azY, int azYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *aL, int aLSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *axL, int axLSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *ayL, int ayLSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *azL, int azLSize)};
%apply (int* INPLACE_ARRAY1, int DIM1){(int *indices, int indicesSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *ii, int iiSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *jj, int jjSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *kk, int kkSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *M, int MSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Mx, int MxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *My, int MySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Mz, int MzSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dxa, int dxaSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dya, int dyaSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *dza, int dzaSize)};
extern void getCoeff(double *aY, int aYSize, 
                    double *axY, int axYSize, 
                    double *ayY, int ayYSize, 
                    double *azY, int azYSize, 
                    double *aL, int aLSize, 
                    double *axL, int axLSize, 
                    double *ayL, int ayLSize, 
                    double *azL, int azLSize, 
                    double dx, double dy, double dz, int P, double kappa);
extern void multipole(double *Lap, int LSize,
                      double *dL, int dLSize,
                      double *Y, int YSize,
                      double *dY, int dYSize,
                      double *M, int MSize,
                      double *Mx, int MxSize,
                      double *My, int MySize,
                      double *Mz, int MzSize,
                      double *dxa, int dxaSize,
                      double *dya, int dyaSize,
                      double *dza, int dzaSize,
                      int P, double kappa, int Nm, double E_hat);
extern void getIndex_arr(int P, int N, 
                        int *indices, int indicesSize,
                        int * ii    , int iiSize,
                        int * jj    , int jjSize,
                        int * kk    , int kkSize);
extern int getIndex(int P, int i, int j, int k);
%clear (double *Lap, int LSize);
%clear (double *dL, int dLSize);
%clear (double *Y, int YSize);
%clear (double *dY, int dYSize);
%clear (double *M, int MSize);
%clear (double *Mx, int MxSize);
%clear (double *My, int MySize);
%clear (double *Mz, int MzSize);
%clear (double *dxa, int dxaSize);
%clear (double *dya, int dyaSize);
%clear (double *dza, int dzaSize);
%clear (double *aY, int aYSize);
%clear (double *axY, int axYSize);
%clear (double *ayY, int ayYSize);
%clear (double *azY, int azYSize);
%clear (double *aL, int aLSize);
%clear (double *axL, int axLSize);
%clear (double *ayL, int ayLSize);
%clear (double *azL, int azLSize);
%clear (double *indices, int indicesSize);
%clear (double *ii, int iiSize);
%clear (double *jj, int jjSize);
%clear (double *kk, int kkSize);
