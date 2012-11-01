#include <cmath>
#include <stdio.h>
#define REAL double

REAL __inline__ power(REAL x, int I)
{
    if (I==0)
        return 1.;

    REAL result=1.;
    for (int i=0; i<I; i++)
    {
        result *= x;
    }
    return result;
}

void P2M(REAL *M, int Msize, REAL *Mx, int Mxsize, REAL *My, int Mysize, REAL *Mz, int Mzsize,
        REAL *x, int xSize, REAL *y, int ySize, REAL *z, int zSize, 
        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize,
        REAL xc, REAL yc, REAL zc, int *I, int Isize, int *J, int Jsize, int *K, int Ksize)
{
    REAL dx, dy, dz, constant;
    for (int i=0; i<Isize; i++)
    {
        for (int j=0; j<xSize; j++)
        {
            dx = xc - x[j];
            dy = yc - y[j];
            dz = zc - z[j];
            constant = power(dx,I[i]) * power(dy,J[i]) * power(dz,K[i]);
            M[i] += m[j] * constant;
            Mx[i] += mx[j] * constant;
            My[i] += my[j] * constant;
            Mz[i] += mz[j] * constant;
        }
    }
}

void M2M(REAL *MP, int MPsize, REAL *MC, int MCsize, REAL dx, REAL dy, REAL dz, 
         int *I, int Isize, int *J, int Jsize, int *K, int Ksize,  
         REAL *cI, int cIsize, REAL *cJ, int cJsize, REAL *cK, int cKsize,  
         int *Imi, int Imisize, int *Jmj, int Jmjsize, int *Kmk, int Kmksize,
         int *index, int indexSize, int *ptr, int ptrSize)
{
    int size, ptr_start, Mptr;
    for (int i=0; i<MPsize; i++)
    {
        ptr_start = ptr[i];
        size      = ptr[i+1] - ptr_start;

        for (int j=0; j<size; j++)
        {
            Mptr = ptr_start + j;
            MP[i] += MC[index[Mptr]]*cI[Mptr]*cJ[Mptr]*cK[Mptr]*power(dx,Imi[Mptr])*power(dy,Jmj[Mptr])*power(dz,Kmk[Mptr]);    
        }
    }
}
