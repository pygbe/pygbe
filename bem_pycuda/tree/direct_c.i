%module direct_c

%{
#define SWIG_FILE_WITH_INIT
extern void direct_c(double *xi, int xiSize, 
                    double *yi, int yiSize, 
                    double *zi, int ziSize, 
                    double *xj, int xjSize, 
                    double *yj, int yjSize, 
                    double *zj, int zjSize, 
                    double *ds, int dsSize, 
                    double *dsx, int dsxSize, 
                    double *dsy, int dsySize, 
                    double *dsz, int dszSize, 
                    double *m, int mSize,
                    double kappa, double eps);
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1){(double *xi, int xiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yi, int yiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zi, int ziSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xj, int xjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yj, int yjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zj, int zjSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *ds, int dsSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dsx, int dsxSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dsy, int dsySize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dsz, int dszSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *m, int mSize)};
extern void direct_c(double *xi, int xiSize, 
                    double *yi, int yiSize, 
                    double *zi, int ziSize, 
                    double *xj, int xjSize, 
                    double *yj, int yjSize, 
                    double *zj, int zjSize, 
                    double *ds, int dsSize, 
                    double *dsx, int dsxSize, 
                    double *dsy, int dsySize, 
                    double *dsz, int dszSize, 
                    double *m, int mSize,
                    double kappa, double eps);

%clear (double *ds, int dsSize);
%clear (double *dsx, int dsxSize);
%clear (double *dsy, int dsySize);
%clear (double *dsz, int dszSize);
%clear (double *xi, int xiSize);
%clear (double *yi, int yiSize);
%clear (double *zi, int ziSize);
%clear (double *xj, int xjSize);
%clear (double *yj, int yjSize);
%clear (double *zj, int zjSize);
%clear (double *m, int mSize);
