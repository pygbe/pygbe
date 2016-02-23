/* 
  Copyright (C) 2013 by Christopher Cooper, Lorena Barba

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/ 

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

void P2M(REAL *M, int Msize, REAL *Md, int Mdsize,
        REAL *x, int xSize, REAL *y, int ySize, REAL *z, int zSize, 
        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize,
        REAL xc, REAL yc, REAL zc, int *I, int Isize, int *J, int Jsize, int *K, int Ksize)
{
    REAL dx, dy, dz, dxI, dyJ, dzK, constant;
    for (int i=0; i<Isize; i++)
    {
        for (int j=0; j<xSize; j++)
        {
            dx = xc - x[j];
            dy = yc - y[j];
            dz = zc - z[j];
            dxI   = power(dx,I[i]);
            dyJ   = power(dy,J[i]);
            dzK   = power(dz,K[i]);
            constant = dxI*dyJ*dzK;
            M[i] += m[j] * constant;
            Md[i] -= mx[j] * I[i]*constant/dx;
            Md[i] -= my[j] * J[i]*constant/dy;
            Md[i] -= mz[j] * K[i]*constant/dz;
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
