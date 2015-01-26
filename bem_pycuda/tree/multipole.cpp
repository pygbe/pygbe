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
#include <iostream>
#define REAL double

int setIndex(int P, int i, int j, int k)
{
    int I=0, ii, jj;
    for (ii=0; ii<i; ii++)
    {
        for (jj=1; jj<P+2-ii; jj++)
        {
            I+=jj;
        }
    }
    for (jj=P+2-j; jj<P+2; jj++)
    {
        I+=jj-i;
    }
    I+=k;

    return I;
}

void getIndex_arr(int P, int N, int *indices, int indicesSize, int *ii, int iiSize, int *jj, int jjSize, int *kk, int kkSize)
{
    for (int iter=0; iter<N; iter++)
        indices[iter] = setIndex(P, ii[iter], jj[iter], kk[iter]);

}

int __inline__ getIndex(int P, int i, int j, int k, int *Index)
{
    return Index[i*(P+1)*(P+1)+j*(P+1)+k];
}

void getCoeff(REAL *a, REAL dx, REAL dy, REAL dz, int *index,
                int Nm, int P, REAL kappa, int LorY)
{

    REAL b[Nm];

    REAL R = sqrt(dx*dx+dy*dy+dz*dz);
    REAL R2 = R*R;
    REAL R3 = R2*R;
    
    int i,j,k,I,Im1x,Im2x,Im1y,Im2y,Im1z,Im2z;
    REAL C,C1,C2,Cb;

    if (LorY==2) // if Yukawa
    {
        b[0] = exp(-kappa*R);
        a[0] = b[0]/R;
    }

    if (LorY==1) // if Laplace
    {
        a[0] = 1/R;
    }

    // Two indices = 0
    I = getIndex(P,1,0,0, index);

    if (LorY==2) // if Yukawa
    {
        b[I]   = -kappa * (dx*a[0]); // 1,0,0
        b[P+1] = -kappa * (dy*a[0]); // 0,1,0
        b[1]   = -kappa * (dz*a[0]); // 0,0,1

        a[I]   = -1/R2*(kappa*dx*b[0]+dx*a[0]);
        a[P+1] = -1/R2*(kappa*dy*b[0]+dy*a[0]);
        a[1]   = -1/R2*(kappa*dz*b[0]+dz*a[0]);
        
    }

    if (LorY==1) // if Laplace
    {
        a[I]   = -dx/R3;
        a[P+1] = -dy/R3;
        a[1]   = -dz/R3;

    }
    
    for (i=2; i<P+1; i++)
    {
        Cb   = -kappa/i;
        C    = 1./(i*R2);
        I    = getIndex(P,i,0,0, index);
        Im1x = getIndex(P,i-1,0,0, index);
        Im2x = getIndex(P,i-2,0,0, index);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + a[Im2x]);
            a[I] = C * ( -kappa*(dx*b[Im1x] + b[Im2x]) -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
        }

        I    = getIndex(P,0,i,0, index);
        Im1y = I-(P+2-i);
        Im2y = Im1y-(P+2-i+1);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + a[Im2y]);
            a[I] = C * ( -kappa*(dy*b[Im1y] + b[Im2y]) -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
        }

        I   = i;
        Im1z = I-1;
        Im2z = I-2;
        
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dz*a[Im1z] + a[Im2z]);
            a[I] = C * ( -kappa*(dz*b[Im1z] + b[Im2z]) -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
        }
    }

    // One index = 0, one = 1 other >=1

    Cb   = -kappa/2;
    I    = getIndex(P,1,1,0, index);
    Im1x = P+1;
    Im1y = I-(P+2-1-1);
    if (LorY==2) // if Yukawa
    {
        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y]);
        a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]) -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
    }

    if (LorY==1) // if Laplace
    {
        a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
    }

    I    = getIndex(P,1,0,1, index);
    Im1x = 1;
    Im1z = I-1;
    if (LorY==2) // if Yukawa
    {
        b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z]);
        a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]) -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
    }

    if (LorY==1) // if Laplace
    {
        a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
    }

    I    = getIndex(P,0,1,1, index);
    Im1y = I-(P+2-1);
    Im1z = I-1;

    if (LorY==2) // if Yukawa
    {
        b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z]);
        a[I] = 1./(2*R2) * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]) -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
    }

    if (LorY==1) // if Laplace
    {
        a[I] = 1./(2*R2) * ( -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
    }

    for (i=2; i<P; i++)
    {
        Cb   = -kappa/(i+1);
        C    = 1./((1+i)*R2);
        I    = getIndex(P,1,i,0, index);
        Im1x = getIndex(P,0,i,0, index);
        Im1y = I-(P+2-i-1);
        Im2y = Im1y-(P+2-i);

        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2y]);
            a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }

        I    = getIndex(P,1,0,i, index);
        Im1x = getIndex(P,0,0,i, index);
        Im1z = I-1;
        Im2z = I-2;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2z]);
            a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }
        
        I    = getIndex(P,0,1,i, index);
        Im1y = I-(P+2-1);
        Im1z = I-1;
        Im2z = I-2;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
            a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }
        
        I    = getIndex(P,i,1,0, index);
        Im1y = I-(P+2-1-i);
        Im1x = getIndex(P,i-1,1,0, index);
        Im2x = getIndex(P,i-2,1,0, index);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + dx*a[Im1x] + a[Im2x]);
            a[I] = C * ( -kappa*(dy*b[Im1y]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }
        
        I    = getIndex(P,i,0,1, index);
        Im1z = I-1;
        Im1x = getIndex(P,i-1,0,1, index);
        Im2x = getIndex(P,i-2,0,1, index);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dz*a[Im1z] + dx*a[Im1x] + a[Im2x]);
            a[I] = C * ( -kappa*(dz*b[Im1z]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }

        I    = getIndex(P,0,i,1, index);
        Im1z = I-1;
        Im1y = I-(P+2-i);
        Im2y = Im1y-(P+2-i+1);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dz*a[Im1z] + dy*a[Im1y] + a[Im2y]);
            a[I] = C * ( -kappa*(dz*b[Im1z]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }
    }

    // One index 0, others >=2
    for (i=2; i<P+1; i++)
    {
        for (j=2; j<P+1-i; j++)
        {
            Cb   = -kappa/(i+j); 
            C    = 1./((i+j)*R2);
            I    = getIndex(P,i,j,0, index);
            Im1x = getIndex(P,i-1,j,0, index);
            Im2x = getIndex(P,i-2,j,0, index);
            Im1y = I-(P+2-j-i);
            Im2y = Im1y-(P+3-j-i);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2x] + a[Im2y]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2x]+b[Im2y]) -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
            }

            I    = getIndex(P,i,0,j, index);
            Im1x = getIndex(P,i-1,0,j, index);
            Im2x = getIndex(P,i-2,0,j, index);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2x]+b[Im2z]) -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
            }

            I    = getIndex(P,0,i,j, index);
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+3-i);
            Im1z = I-1;
            Im2z = I-2; 
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
            }
        }
    }

    if (P>2)
    {
        // Two index = 1, other>=1
        Cb   = -kappa/3;
        I    = getIndex(P,1,1,1, index);
        Im1x = getIndex(P,0,1,1, index);
        Im1y = getIndex(P,1,0,1, index);
        Im1y = I-(P);
        Im1z = I-1;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z]);
            a[I] = 1/(3*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]) -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1/(3*R2) * ( -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
        }

        for (i=2; i<P-1; i++)
        {
            Cb   = -kappa/(2+i);
            C    = 1./((i+2)*R2);
            I    = getIndex(P,i,1,1, index);
            Im1x = getIndex(P,i-1,1,1, index);
            Im1y = I-(P+2-i-1);
            Im1z = I-1;
            Im2x = getIndex(P,i-2,1,1, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
            }

            I    = getIndex(P,1,i,1, index);
            Im1x = getIndex(P,0,i,1, index);
            Im1y = I-(P+2-i-1);
            Im2y = Im1y-(P+3-i-1);
            Im1z = I-1 ;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
            }


            I    = getIndex(P,1,1,i, index);
            Im1x = getIndex(P,0,1,i, index);
            Im1y = I-(P);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
            }
        }
    }

    // One index = 1, others >=2
    if (P>4)
    {
        for (i=2; i<P-2; i++)
        {
            for (j=2; j<P-i; j++)
            {
                Cb = -kappa/(1+i+j);
                C  = 1./((1+i+j)*R2);
                C1 = -(2.*(1+i+j)-1); 
                C2 = (i+j);
                I    = getIndex(P,1,i,j, index);
                Im1x = getIndex(P,0,i,j, index);
                Im1y = I-(P+2-1-i);
                Im2y = Im1y-(P+3-1-i);
                Im1z = I-1;
                Im2z = I-2; 
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                }

                I    = getIndex(P,i,1,j, index);
                Im1x = getIndex(P,i-1,1,j, index);
                Im1y = I-(P+2-i-1);
                Im2x = getIndex(P,i-2,1,j, index);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                }
                
                I    = getIndex(P,i,j,1, index);
                Im1x = getIndex(P,i-1,j,1, index);
                Im2x = getIndex(P,i-2,j,1, index);
                Im1y = I-(P+2-i-j);
                Im2y = Im1y-(P+3-i-j); 
                Im1z = I-1;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                }
            }
        }
    }

    // All indices >= 2
    if (P>5)
    {
        for (i=2;i<P-3;i++)
        {
            for (j=2;j<P-1-i;j++)
            {
                for (k=2;k<P+1-i-j;k++)
                {
                    Cb = -kappa/(i+j+k);
                    C  = 1./((i+j+k)*R2);
                    C1 = -(2.*(i+j+k)-1); 
                    C2 = i+j+k-1.;
                    I    = getIndex(P,i,j,k, index);
                    Im1x = getIndex(P,i-1,j,k, index);
                    Im2x = getIndex(P,i-2,j,k, index);
                    Im1y = I-(P+2-i-j);
                    Im2y = Im1y-(P+3-i-j); 
                    Im1z = I-1; 
                    Im2z = I-2; 
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y] + a[Im2z]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                    }

                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                    }
                }
            }
        }
    }
}

void getCoeff_shift(REAL *ax, REAL *ay, REAL *az, REAL dx, REAL dy, REAL dz, int *index,
                int Nm, int P, REAL kappa, int LorY)
{

    REAL b[Nm], a[Nm];

    REAL R = sqrt(dx*dx+dy*dy+dz*dz);
    REAL R2 = R*R;
    REAL R3 = R2*R;
    
    int i,j,k,I,Im1x,Im2x,Im1y,Im2y,Im1z,Im2z;
    REAL C,C1,C2,Cb;

    if (LorY==2) // if Yukawa
    {
        b[0] = exp(-kappa*R);
        a[0] = b[0]/R;
    }

    if (LorY==1) // if Laplace
    {
        a[0] = 1/R;
    }

    // Two indices = 0
    I = getIndex(P,1,0,0, index);

    if (LorY==2) // if Yukawa
    {
        b[I]   = -kappa * (dx*a[0]); // 1,0,0
        b[P+1] = -kappa * (dy*a[0]); // 0,1,0
        b[1]   = -kappa * (dz*a[0]); // 0,0,1

        a[I]   = -1/R2*(kappa*dx*b[0]+dx*a[0]);
        a[P+1] = -1/R2*(kappa*dy*b[0]+dy*a[0]);
        a[1]   = -1/R2*(kappa*dz*b[0]+dz*a[0]);
        
    }

    if (LorY==1) // if Laplace
    {
        a[I]   = -dx/R3;
        a[P+1] = -dy/R3;
        a[1]   = -dz/R3;

    }
    
    ax[0] = a[I];
    ay[0] = a[P+1];
    az[0] = a[1];

    for (i=2; i<P+1; i++)
    {
        Cb   = -kappa/i;
        C    = 1./(i*R2);
        I    = getIndex(P,i,0,0, index);
        Im1x = getIndex(P,i-1,0,0, index);
        Im2x = getIndex(P,i-2,0,0, index);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + a[Im2x]);
            a[I] = C * ( -kappa*(dx*b[Im1x] + b[Im2x]) -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
        }

        ax[Im1x] = a[I]*i;

        I    = getIndex(P,0,i,0, index);
        Im1y = I-(P+2-i);
        Im2y = Im1y-(P+2-i+1);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + a[Im2y]);
            a[I] = C * ( -kappa*(dy*b[Im1y] + b[Im2y]) -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*i-1)*dy*a[Im1y] - (i-1)*a[Im2y] );
        }

        ay[Im1y] = a[I]*i;

        I   = i;
        Im1z = I-1;
        Im2z = I-2;
        
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dz*a[Im1z] + a[Im2z]);
            a[I] = C * ( -kappa*(dz*b[Im1z] + b[Im2z]) -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*i-1)*dz*a[Im1z] - (i-1)*a[Im2z] );
        }

        az[Im1z] = a[I]*i;
    }

    // One index = 0, one = 1 other >=1

    Cb   = -kappa/2;
    I    = getIndex(P,1,1,0, index);
    Im1x = P+1;
    Im1y = I-(P+2-1-1);
    if (LorY==2) // if Yukawa
    {
        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y]);
        a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]) -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
    }

    if (LorY==1) // if Laplace
    {
        a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dy*a[Im1y]) );
    }

    ax[Im1x] = a[I];
    ay[Im1y] = a[I];

    I    = getIndex(P,1,0,1, index);
    Im1x = 1;
    Im1z = I-1;
    if (LorY==2) // if Yukawa
    {
        b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z]);
        a[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]) -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
    }

    if (LorY==1) // if Laplace
    {
        a[I] = 1./(2*R2) * ( -(2*2-1)*(dx*a[Im1x]+dz*a[Im1z]) );
    }

    ax[Im1x] = a[I];
    az[Im1z] = a[I];

    I    = getIndex(P,0,1,1, index);
    Im1y = I-(P+2-1);
    Im1z = I-1;

    if (LorY==2) // if Yukawa
    {
        b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z]);
        a[I] = 1./(2*R2) * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]) -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
    }

    if (LorY==1) // if Laplace
    {
        a[I] = 1./(2*R2) * ( -(2*2-1)*(dy*a[Im1y]+dz*a[Im1z]) );
    }

    ay[Im1y] = a[I];
    az[Im1z] = a[I];

    for (i=2; i<P; i++)
    {
        Cb   = -kappa/(i+1);
        C    = 1./((1+i)*R2);
        I    = getIndex(P,1,i,0, index);
        Im1x = getIndex(P,0,i,0, index);
        Im1y = I-(P+2-i-1);
        Im2y = Im1y-(P+2-i);

        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2y]);
            a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }

        ax[Im1x] = a[I];
        ay[Im1y] = a[I]*i;

        I    = getIndex(P,1,0,i, index);
        Im1x = getIndex(P,0,0,i, index);
        Im1z = I-1;
        Im2z = I-2;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2z]);
            a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dx*a[Im1x]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }
        
        ax[Im1x] = a[I];
        az[Im1z] = a[I]*i;

        I    = getIndex(P,0,1,i, index);
        Im1y = I-(P+2-1);
        Im1z = I-1;
        Im2z = I-2;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
            a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dz*a[Im1z]) - (1+i-1)*(a[Im2z]) );
        }
        
        ay[Im1y] = a[I];
        az[Im1z] = a[I]*i;

        I    = getIndex(P,i,1,0, index);
        Im1y = I-(P+2-1-i);
        Im1x = getIndex(P,i-1,1,0, index);
        Im2x = getIndex(P,i-2,1,0, index);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dy*a[Im1y] + dx*a[Im1x] + a[Im2x]);
            a[I] = C * ( -kappa*(dy*b[Im1y]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }
        
        ay[Im1y] = a[I];
        ax[Im1x] = a[I]*i;

        I    = getIndex(P,i,0,1, index);
        Im1z = I-1;
        Im1x = getIndex(P,i-1,0,1, index);
        Im2x = getIndex(P,i-2,0,1, index);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dz*a[Im1z] + dx*a[Im1x] + a[Im2x]);
            a[I] = C * ( -kappa*(dz*b[Im1z]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
        }

        az[Im1z] = a[I];
        ax[Im1x] = a[I]*i;

        I    = getIndex(P,0,i,1, index);
        Im1z = I-1;
        Im1y = I-(P+2-i);
        Im2y = Im1y-(P+2-i+1);
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dz*a[Im1z] + dy*a[Im1y] + a[Im2y]);
            a[I] = C * ( -kappa*(dz*b[Im1z]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }
        if (LorY==1) // if Laplace
        {
            a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dy*a[Im1y]) - (1+i-1)*(a[Im2y]) );
        }

        az[Im1z] = a[I];
        ay[Im1y] = a[I]*i;

    }

    // One index 0, others >=2
    for (i=2; i<P+1; i++)
    {
        for (j=2; j<P+1-i; j++)
        {
            Cb   = -kappa/(i+j); 
            C    = 1./((i+j)*R2);
            I    = getIndex(P,i,j,0, index);
            Im1x = getIndex(P,i-1,j,0, index);
            Im2x = getIndex(P,i-2,j,0, index);
            Im1y = I-(P+2-j-i);
            Im2y = Im1y-(P+3-j-i);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + a[Im2x] + a[Im2y]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2x]+b[Im2y]) -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dy*a[Im1y]) -(i+j-1)*(a[Im2x]+a[Im2y]) );
            }

            ax[Im1x] = a[I]*i;
            ay[Im1y] = a[I]*j;

            I    = getIndex(P,i,0,j, index);
            Im1x = getIndex(P,i-1,0,j, index);
            Im2x = getIndex(P,i-2,0,j, index);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2x]+b[Im2z]) -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+j)-1)*(dx*a[Im1x]+dz*a[Im1z]) -(i+j-1)*(a[Im2x]+a[Im2z]) );
            }

            ax[Im1x] = a[I]*i;
            az[Im1z] = a[I]*j;

            I    = getIndex(P,0,i,j, index);
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+3-i);
            Im1z = I-1;
            Im2z = I-2; 
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                a[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+j)-1)*(dy*a[Im1y]+dz*a[Im1z]) -(i+j-1)*(a[Im2y]+a[Im2z]) );
            }

            ay[Im1y] = a[I]*i;
            az[Im1z] = a[I]*j;

        }
    }

    if (P>2)
    {
        // Two index = 1, other>=1
        Cb   = -kappa/3;
        I    = getIndex(P,1,1,1, index);
        Im1x = getIndex(P,0,1,1, index);
        Im1y = getIndex(P,1,0,1, index);
        Im1y = I-(P);
        Im1z = I-1;
        if (LorY==2) // if Yukawa
        {
            b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z]);
            a[I] = 1/(3*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]) -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
        }

        if (LorY==1) // if Laplace
        {
            a[I] = 1/(3*R2) * ( -5*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) );
        }

        ax[Im1x] = a[I];
        ay[Im1y] = a[I];
        az[Im1z] = a[I];

        for (i=2; i<P-1; i++)
        {
            Cb   = -kappa/(2+i);
            C    = 1./((i+2)*R2);
            I    = getIndex(P,i,1,1, index);
            Im1x = getIndex(P,i-1,1,1, index);
            Im1y = I-(P+2-i-1);
            Im1z = I-1;
            Im2x = getIndex(P,i-2,1,1, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
            }

            ax[Im1x] = a[I]*i;
            ay[Im1y] = a[I];
            az[Im1z] = a[I];

            I    = getIndex(P,1,i,1, index);
            Im1x = getIndex(P,0,i,1, index);
            Im1y = I-(P+2-i-1);
            Im2y = Im1y-(P+3-i-1);
            Im1z = I-1 ;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2y]) );
            }

            ax[Im1x] = a[I];
            ay[Im1y] = a[I]*i;
            az[Im1z] = a[I];

            I    = getIndex(P,1,1,i, index);
            Im1x = getIndex(P,0,1,i, index);
            Im1y = I-(P);
            Im1z = I-1;
            Im2z = I-2;
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2z]);
                a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2z]) );
            }

            ax[Im1x] = a[I];
            ay[Im1y] = a[I];
            az[Im1z] = a[I]*i;

        }
    }

    // One index = 1, others >=2
    if (P>4)
    {
        for (i=2; i<P-2; i++)
        {
            for (j=2; j<P-i; j++)
            {
                Cb = -kappa/(1+i+j);
                C  = 1./((1+i+j)*R2);
                C1 = -(2.*(1+i+j)-1); 
                C2 = (i+j);
                I    = getIndex(P,1,i,j, index);
                Im1x = getIndex(P,0,i,j, index);
                Im1y = I-(P+2-1-i);
                Im2y = Im1y-(P+3-1-i);
                Im1z = I-1;
                Im2z = I-2; 
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2y] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2y]+a[Im2z]) );
                }
        
                ax[Im1x] = a[I];
                ay[Im1y] = a[I]*i;
                az[Im1z] = a[I]*j;

                I    = getIndex(P,i,1,j, index);
                Im1x = getIndex(P,i-1,1,j, index);
                Im1y = I-(P+2-i-1);
                Im2x = getIndex(P,i-2,1,j, index);
                Im1z = I-1;
                Im2z = I-2;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2z]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2z]) );
                }
                
                ax[Im1x] = a[I]*i;
                ay[Im1y] = a[I];
                az[Im1z] = a[I]*j;

                I    = getIndex(P,i,j,1, index);
                Im1x = getIndex(P,i-1,j,1, index);
                Im2x = getIndex(P,i-2,j,1, index);
                Im1y = I-(P+2-i-j);
                Im2y = Im1y-(P+3-i-j); 
                Im1z = I-1;
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]) );
                }

                ax[Im1x] = a[I]*i;
                ay[Im1y] = a[I]*j;
                az[Im1z] = a[I];

            }
        }
    }

    // All indices >= 2
    if (P>5)
    {
        for (i=2;i<P-3;i++)
        {
            for (j=2;j<P-1-i;j++)
            {
                for (k=2;k<P+1-i-j;k++)
                {
                    Cb = -kappa/(i+j+k);
                    C  = 1./((i+j+k)*R2);
                    C1 = -(2.*(i+j+k)-1); 
                    C2 = i+j+k-1.;
                    I    = getIndex(P,i,j,k, index);
                    Im1x = getIndex(P,i-1,j,k, index);
                    Im2x = getIndex(P,i-2,j,k, index);
                    Im1y = I-(P+2-i-j);
                    Im2y = Im1y-(P+3-i-j); 
                    Im1z = I-1; 
                    Im2z = I-2; 
                    if (LorY==2) // if Yukawa
                    {
                        b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x] + a[Im2y] + a[Im2z]);
                        a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]+b[Im2z]) + C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                    }

                    if (LorY==1) // if Laplace
                    {
                        a[I] = C * ( C1*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - C2*(a[Im2x]+a[Im2y]+a[Im2z]) );
                    }

                    ax[Im1x] = a[I]*i;
                    ay[Im1y] = a[I]*j;
                    az[Im1z] = a[I]*k;
                }
            }
        }
    }
}

void multipole_c(REAL *K_aux , int K_auxSize, 
                 REAL *V_aux , int V_auxSize,
                   REAL *M , int MSize, 
                   REAL *Md, int MdSize, 
                   REAL *dx, int dxSize, 
                   REAL *dy, int dySize, 
                   REAL *dz, int dzSize,
                   int *index, int indexSize,
                   int P, REAL kappa, int Nm, int LorY)
{
    REAL a[Nm];

    for (int i=0; i<K_auxSize; i++)
    {   
        for (int ii=0; ii<Nm; ii++)
        {   
            a[ii] = 0.; 
        }   

        getCoeff(a, dx[i], dy[i], dz[i], 
                 index,  Nm, P, kappa, LorY);

        for (int j=0; j<Nm; j++)
        {   
            V_aux[i] += a[j]*M[j];
            K_aux[i] += a[j]*Md[j];
        }   

    }   
}

void multipole_sort(REAL *K_aux , int K_auxSize, 
                    REAL *V_aux , int V_auxSize,
                    int *offTar, int offTarSize,
                    int *sizeTar, int sizeTarSize,
                    int *offMlt, int offMltSize,
                    REAL *M , int MSize, 
                    REAL *Md, int MdSize, 
                    REAL *xi, int xiSize, 
                    REAL *yi, int yiSize, 
                    REAL *zi, int ziSize,
                    REAL *xc, int xcSize, 
                    REAL *yc, int ycSize, 
                    REAL *zc, int zcSize,
                    int *index, int indexSize,
                    int P, REAL kappa, int Nm, int LorY)
{
    REAL a[Nm], dx, dy, dz;
    int CI_begin, CI_end, CJ_begin, CJ_end;

    for(int CI=0; CI<offTarSize; CI++)
    {
        CI_begin = offTar[CI];
        CI_end   = offTar[CI] + sizeTar[CI];
        CJ_begin = offMlt[CI];
        CJ_end   = offMlt[CI+1];

        for(int CJ=CJ_begin; CJ<CJ_end; CJ++)
        {
            for (int i=CI_begin; i<CI_end; i++)
            {   
                for (int ii=0; ii<Nm; ii++)
                {   
                    a[ii] = 0.; 
                }   

                dx = xi[i] - xc[CJ];
                dy = yi[i] - yc[CJ];
                dz = zi[i] - zc[CJ];

                getCoeff(a, dx, dy, dz, index,  
                        Nm, P, kappa, LorY);

                for (int j=0; j<Nm; j++)
                {   
                    V_aux[i] += a[j]*M[CJ*Nm+j];
                    K_aux[i] += a[j]*Md[CJ*Nm+j];
                } 
            }   
        }
    }
}

void multipoleKt_sort(REAL *Ktx_aux , int Ktx_auxSize, 
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
{
    REAL ax[Nm], ay[Nm], az[Nm], dx, dy, dz;
    int CI_begin, CI_end, CJ_begin, CJ_end;

    for(int CI=0; CI<offTarSize; CI++)
    {
        CI_begin = offTar[CI];
        CI_end   = offTar[CI] + sizeTar[CI];
        CJ_begin = offMlt[CI];
        CJ_end   = offMlt[CI+1];

        for(int CJ=CJ_begin; CJ<CJ_end; CJ++)
        {
            for (int i=CI_begin; i<CI_end; i++)
            {   
                for (int ii=0; ii<Nm; ii++)
                {   
                    ax[ii] = 0.; 
                    ay[ii] = 0.; 
                    az[ii] = 0.; 
                }   

                dx = xi[i] - xc[CJ];
                dy = yi[i] - yc[CJ];
                dz = zi[i] - zc[CJ];

                getCoeff_shift(ax, ay, az, dx, dy, dz, index,  
                        Nm, P, kappa, LorY);

                for (int j=0; j<Nm; j++)
                {   
                    Ktx_aux[i] += ax[j]*M[CJ*Nm+j];
                    Kty_aux[i] += ay[j]*M[CJ*Nm+j];
                    Ktz_aux[i] += az[j]*M[CJ*Nm+j];
                } 
            }   
        }
    }
}
