%module calculateMultipoles

%{
#define SWIG_FILE_WITH_INIT
extern void P2M(double *M, int Msize,
                double *Md, int Mdsize,
                double *x, int xSize, 
                double *y, int ySize, 
                double *z, int zSize, 
                double *m, int mSize, 
                double *mx, int mxSize, 
                double *my, int mySize, 
                double *mz, int mzSize, 
                double xc, double yc, double zc,
                int *I, int Isize,
                int *J, int Jsize,
                int *K, int Ksize);

extern void M2M(double *MP, int MPsize,
                double *MC, int MCsize,
                double dx, double dy, double dz,
                int *I, int Isize, int *J, int Jsize, int *K, int Ksize,
                double *cI, int cIsize, double *cJ, int cJsize, double *cK, int cKsize,
                int *Imi, int Imisize, int *Jmj, int Jmjsize, int *Kmk, int Kmksize,
                int *index, int indexSize, int *ptr, int ptrSize);
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1){(double *M, int Msize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *Md, int Mdsize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *MP, int MPsize)};
%apply (double* IN_ARRAY1, int DIM1){(double *MC, int MCsize)};
%apply (double* IN_ARRAY1, int DIM1){(double *x, int xSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *y, int ySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *z, int zSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *m, int mSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mx, int mxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *my, int mySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mz, int mzSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *I, int Isize)};
%apply (int* IN_ARRAY1, int DIM1){(int *J, int Jsize)};
%apply (int* IN_ARRAY1, int DIM1){(int *K, int Ksize)};
%apply (double* IN_ARRAY1, int DIM1){(double *cI, int cIsize)};
%apply (double* IN_ARRAY1, int DIM1){(double *cJ, int cJsize)};
%apply (double* IN_ARRAY1, int DIM1){(double *cK, int cKsize)};
%apply (int* IN_ARRAY1, int DIM1){(int *Imi, int Imisize)};
%apply (int* IN_ARRAY1, int DIM1){(int *Jmj, int Jmjsize)};
%apply (int* IN_ARRAY1, int DIM1){(int *Kmk, int Kmksize)};
%apply (int* IN_ARRAY1, int DIM1){(int *index, int indexSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *ptr, int ptrSize)};
extern void P2M(double *M, int Msize,
                double *Md, int Mdsize,
                double *x, int xSize, 
                double *y, int ySize, 
                double *z, int zSize, 
                double *m, int mSize, 
                double *mx, int mxSize, 
                double *my, int mySize, 
                double *mz, int mzSize, 
                double xc, double yc, double zc,
                int *I, int Isize,
                int *J, int Jsize,
                int *K, int Ksize);

extern void M2M(double *MP, int MPsize,
                double *MC, int MCsize,
                double dx, double dy, double dz,
                int *I, int Isize, int *J, int Jsize, int *K, int Ksize,
                double *cI, int cIsize, double *cJ, int cJsize, double *cK, int cKsize,
                int *Imi, int Imisize, int *Jmj, int Jmjsize, int *Kmk, int Kmksize,
                int *index, int indexSize, int *ptr, int ptrSize);

%clear (double *M, int Msize);
%clear (double *Md, int Mdsize);
%clear (double *MP, int MPsize);
%clear (double *MC, int MCsize);
%clear (double *x, int xSize);
%clear (double *y, int ySize);
%clear (double *z, int zSize);
%clear (double *m, int mSize);
%clear (double *mx, int mxSize);
%clear (double *my, int mySize);
%clear (double *mz, int mzSize);
%clear (int *I, int Isize);
%clear (int *J, int Jsize);
%clear (int *K, int Ksize);
%clear (double *cI, int cIsize);
%clear (double *cJ, int cJsize);
%clear (double *cK, int cKsize);
%clear (int *Imi, int Imisize);
%clear (int *Jmj, int Jmjsize);
%clear (int *Kmk, int Kmksize);
%clear (int *index, int indexSize);
%clear (int *ptr, int ptrSize);
