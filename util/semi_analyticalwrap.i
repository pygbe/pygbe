%module semi_analyticalwrap

%{
#define SWIG_FILE_WITH_INIT
extern void SA_wrap_arr(double *y, int ySize, 
                    double *x, int xSize, 
                    double *phi_Y, int phi_YSize, 
                    double *dphi_Y, int dphi_YSize, 
                    double *phi_L, int phi_LSize, 
                    double *dphi_L, int dphi_LSize, 
                    double kappa, 
                    int    *same, int sameSize, 
                    double *xk, int xkSize, 
                    double *wk, int wkSize);

extern void P2P_c(double *MY, int MYSize, double *dMY, int dMYSize, double *ML, int MLSize, double *dML, int dMLSize,
        double *triangle, int triangleSize, int *tri, int triSize, int *k, int kSize, 
        double *xi, int xiSize, double *yi, int yiSize, double *zi, int ziSize,
        double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, double *s_zj, int s_zjSize,
        double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mclean, int mcleanSize, int *targets, int targetsSize, 
        double *Area, int AreaSize, double *xk, int xkSize, double *wk, int wkSize, 
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1){(double *phi_Y, int phi_YSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dphi_Y, int dphi_YSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *phi_L, int phi_LSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dphi_L, int dphi_LSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *same, int sameSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xk, int xkSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *wk, int wkSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *y, int ySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *x, int xSize)};
extern void SA_wrap_arr(double *y, int ySize, 
                    double *x, int xSize, 
                    double *phi_Y, int phi_YSize, 
                    double *dphi_Y, int dphi_YSize, 
                    double *phi_L, int phi_LSize, 
                    double *dphi_L, int dphi_LSize, 
                    double kappa, 
                    int    *same, int sameSize, 
                    double *xk, int xkSize, 
                    double *wk, int wkSize);

%clear (double *phi_Y, int phi_YSize);
%clear (double *dphi_Y, int dphi_YSize);
%clear (double *phi_L, int phi_LSize);
%clear (double *dphi_L, int dphi_LSize);
%clear (int *same, int sameSize);
%clear (double *y, int ySize);
%clear (double *x, int xSize);
%clear (double *xk, int xkSize);
%clear (double *wk, int wkSize);



%apply (double* INPLACE_ARRAY1, int DIM1){(double *MY, int MYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dMY, int dMYSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *ML, int MLSize)};
%apply (double* INPLACE_ARRAY1, int DIM1){(double *dML, int dMLSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *triangle, int triangleSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *tri, int triSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *k, int kSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *s_xj, int s_xjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *s_yj, int s_yjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *s_zj, int s_zjSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xi, int xiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yi, int yiSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zi, int ziSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xt, int xtSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *yt, int ytSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *zt, int ztSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *m, int mSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mx, int mxSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *my, int mySize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mz, int mzSize)};
%apply (int* IN_ARRAY1, int DIM1){(int *targets, int targetsSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *Area, int AreaSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *xk, int xkSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *wk, int wkSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *mclean, int mcleanSize)};
%apply (double* IN_ARRAY1, int DIM1){(double *aux, int auxSize)};
extern void P2P_c(double *MY, int MYSize, double *dMY, int dMYSize, double *ML, int MLSize, double *dML, int dMLSize,
        double *triangle, int triangleSize, int *tri, int triSize, int *k, int kSize, 
        double *xi, int xiSize, double *yi, int yiSize, double *zi, int ziSize,
        double *s_xj, int s_xjSize, double *s_yj, int s_yjSize, double *s_zj, int s_zjSize,
        double *xt, int xtSize, double *yt, int ytSize, double *zt, int ztSize,
        double *m, int mSize, double *mx, int mxSize, double *my, int mySize, double *mz, int mzSize, double *mclean, int mcleanSize, int *targets, int targetsSize,
        double *Area, int AreaSize, double *xk, int xkSize, double *wk, int wkSize, 
        double kappa, double threshold, double eps, double w0, double *aux, int auxSize);
%clear (double *MY, int MYSize); 
%clear (double *dMY, int dMYSize); 
%clear (double *ML, int MLSize); 
%clear (double *dML, int dMLSize); 
%clear (double *triangle, int triangleSize); 
%clear (int *tri, int triSize); 
%clear (int *k, int kSize); 
%clear (double *s_xj, int s_xjSize); 
%clear (double *s_yj, int s_yjSize); 
%clear (double *s_zj, int s_zjSize); 
%clear (double *xt, int xtSize); 
%clear (double *yt, int ytSize); 
%clear (double *zt, int ztSize); 
%clear (double *xi, int xiSize); 
%clear (double *yi, int yiSize); 
%clear (double *zi, int ziSize); 
%clear (double *m, int mSize); 
%clear (double *mx, int mxSize); 
%clear (double *my, int mySize); 
%clear (double *mz, int mzSize); 
%clear (int *targets, int targetsSize); 
%clear (double *Area, int AreaSize); 
%clear (double *xk, int xkSize); 
%clear (double *wk, int wkSize); 
%clear (double *mclean, int mcleanSize); 
%clear (double *aux, int auxSize); 
