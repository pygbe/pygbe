'''
    Copyright (C) 2011 by Christopher Cooper, Lorena Barba
  
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
'''
from pycuda.compiler import SourceModule

def kernels(BSZ, Nm, xkSize):
    
    mod = SourceModule( """

    #define REAL double
    #define BSZ %(blocksize)d
    #define Nm  %(Nmult)d
    #define xkSize %(K_1D)d

	// Find multipole of order i,j,k in 1D array of multipoles
    __device__ int getIndex(int P, int i, int j, int k)
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

	// Recursive relation to find the Taylor coefficients for Laplace and Yukawa for potential and force in x, y, z
    __device__ void getCoeff(REAL *aL, REAL *axL, REAL *ayL, REAL *azL, 
                             REAL *aY, REAL *axY, REAL *ayY, REAL *azY,
                             REAL dx, REAL dy, REAL dz, REAL kappa, int P)
	// aY    : coefficients for Yukawa potential
	// axY   : coefficients for Yukawa force in x direction
	// ayY   : coefficients for Yukawa force in y direction
	// azY   : coefficients for Yukawa force in z direction
	// aL    : coefficients for Laplace potential
	// axL   : coefficients for Laplace force in x direction
	// ayL   : coefficients for Laplace force in y direction
	// azL   : coefficients for Laplace force in z direction
    {
        REAL b[Nm];
        REAL R = sqrt(dx*dx+dy*dy+dz*dz);
        REAL R2 = R*R;
        REAL R3 = R2*R;
        
        int i,j,k,I,Im1x,Im2x,Im1y,Im2y,Im1z,Im2z;
        REAL C,C1,C2,Cb;

        // First coefficient
        b[0] = exp(-kappa*R);
        aY[0] = b[0]/R;
        aL[0] = 1/R;

        // Two indices = 0
        I = getIndex(P,1,0,0);
        b[I]   = -kappa * (dx*aY[0]); // 1,0,0
        b[P+1] = -kappa * (dy*aY[0]); // 0,1,0
        b[1]   = -kappa * (dz*aY[0]); // 0,0,1

        aY[I]   = -1/R2*(kappa*dx*b[0]+dx*aY[0]);
        aY[P+1] = -1/R2*(kappa*dy*b[0]+dy*aY[0]);
        aY[1]   = -1/R2*(kappa*dz*b[0]+dz*aY[0]);
        
        axY[0]  = aY[I]; 
        ayY[0]  = aY[P+1]; 
        azY[0]  = aY[1]; 

        aL[I]   = -dx/R3;
        aL[P+1] = -dy/R3;
        aL[1]   = -dz/R3;

        axL[0]  = -dx/R3; 
        ayL[0]  = -dy/R3; 
        azL[0]  = -dz/R3; 

        for (i=2; i<P+1; i++)
        {   
            Cb   = -kappa/i;
            C    = 1./(i*R2);
            I    = getIndex(P,i,0,0);
            Im1x = getIndex(P,i-1,0,0);
            Im2x = getIndex(P,i-2,0,0);
            b[I] = Cb * (dx*aY[Im1x] + aY[Im2x]);
            aY[I] = C * ( -kappa*(dx*b[Im1x] + b[Im2x]) -(2*i-1)*dx*aY[Im1x] - (i-1)*aY[Im2x] );
            axY[Im1x] = aY[I]*i;
            aL[I] = C * ( -(2*i-1)*dx*aL[Im1x] - (i-1)*aL[Im2x] );
            axL[Im1x] = aL[I]*i;


            I    = getIndex(P,0,i,0);
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+2-i+1);
            b[I] = Cb * (dy*aY[Im1y] + aY[Im2y]);
            aY[I] = C * ( -kappa*(dy*b[Im1y] + b[Im2y]) -(2*i-1)*dy*aY[Im1y] - (i-1)*aY[Im2y] );
            ayY[Im1y] = aY[I]*i;
            aL[I] = C * ( -(2*i-1)*dy*aL[Im1y] - (i-1)*aL[Im2y] );
            ayL[Im1y] = aL[I]*i;


            I   = i;
            Im1z = I-1;
            Im2z = I-2;
            b[I] = Cb * (dz*aY[Im1z] + aY[Im2z]);
            aY[I] = C * ( -kappa*(dz*b[Im1z] + b[Im2z]) -(2*i-1)*dz*aY[Im1z] - (i-1)*aY[Im2z] );
            azY[Im1z] = aY[I]*i;
            aL[I] = C * ( -(2*i-1)*dz*aL[Im1z] - (i-1)*aL[Im2z] );
            azL[Im1z] = aL[I]*i;
        }

        // One index = 0, one = 1 other >=1

        Cb   = -kappa/2;
        I    = getIndex(P,1,1,0);
        Im1x = P+1;
        Im1y = I-(P+2-1-1);
        b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y]);
        aY[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]) -(2*2-1)*(dx*aY[Im1x]+dy*aY[Im1y]) );
        axY[Im1x] = aY[I];
        ayY[Im1y] = aY[I];
        aL[I] = 1./(2*R2) * ( -(2*2-1)*(dx*aL[Im1x]+dy*aL[Im1y]) );
        axL[Im1x] = aL[I];
        ayL[Im1y] = aL[I];

        I    = getIndex(P,1,0,1);
        Im1x = 1;
        Im1z = I-1;
        b[I] = Cb * (dx*aY[Im1x] + dz*aY[Im1z]);
        aY[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]) -(2*2-1)*(dx*aY[Im1x]+dz*aY[Im1z]) );
        axY[Im1x] = aY[I];
        azY[Im1z] = aY[I];
        aL[I] = 1./(2*R2) * ( -(2*2-1)*(dx*aL[Im1x]+dz*aL[Im1z]) );
        axL[Im1x] = aL[I];
        azL[Im1z] = aL[I];

        I    = getIndex(P,0,1,1);
        Im1y = I-(P+2-1);
        Im1z = I-1;
        b[I] = Cb * (dy*aY[Im1y] + dz*aY[Im1z]);
        aY[I] = 1./(2*R2) * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]) -(2*2-1)*(dy*aY[Im1y]+dz*aY[Im1z]) );
        ayY[Im1y] = aY[I];
        azY[Im1z] = aY[I];
        aL[I] = 1./(2*R2) * ( -(2*2-1)*(dy*aL[Im1y]+dz*aL[Im1z]) );
        ayL[Im1y] = aL[I];
        azL[Im1z] = aL[I];

        for (i=2; i<P; i++)
        {
            Cb   = -kappa/(i+1);
            C    = 1./((1+i)*R2);
            I    = getIndex(P,1,i,0);
            Im1x = getIndex(P,0,i,0);
            Im1y = I-(P+2-i-1);
            Im2y = Im1y-(P+2-i);
            b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + aY[Im2y]);
            aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dx*aY[Im1x]+dy*aY[Im1y]) - (1+i-1)*(aY[Im2y]) );
            axY[Im1x] = aY[I];
            ayY[Im1y] = aY[I]*i;
            aL[I] = C * ( -(2*(1+i)-1)*(dx*aL[Im1x]+dy*aL[Im1y]) - (1+i-1)*(aL[Im2y]) );
            axL[Im1x] = aL[I];
            ayL[Im1y] = aL[I]*i;

            I    = getIndex(P,1,0,i);
            Im1x = getIndex(P,0,0,i);
            Im1z = I-1;
            Im2z = I-2;
            b[I] = Cb * (dx*aY[Im1x] + dz*aY[Im1z] + aY[Im2z]);
            aY[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dx*aY[Im1x]+dz*aY[Im1z]) - (1+i-1)*(aY[Im2z]) );
            axY[Im1x] = aY[I];
            azY[Im1z] = aY[I]*i;
            aL[I] = C * ( -(2*(1+i)-1)*(dx*aL[Im1x]+dz*aL[Im1z]) - (1+i-1)*(aL[Im2z]) );
            axL[Im1x] = aL[I];
            azL[Im1z] = aL[I]*i;

            I    = getIndex(P,0,1,i);
            Im1y = I-(P+2-1);
            Im1z = I-1;
            Im2z = I-2;
            b[I] = Cb * (dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2z]);
            aY[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dy*aY[Im1y]+dz*aY[Im1z]) - (1+i-1)*(aY[Im2z]) );
            ayY[Im1y] = aY[I];
            azY[Im1z] = aY[I]*i;
            aL[I] = C * ( -(2*(1+i)-1)*(dy*aL[Im1y]+dz*aL[Im1z]) - (1+i-1)*(aL[Im2z]) );
            ayL[Im1y] = aL[I];
            azL[Im1z] = aL[I]*i;

            I    = getIndex(P,i,1,0);
            Im1y = I-(P+2-1-i);
            Im1x = getIndex(P,i-1,1,0);
            Im2x = getIndex(P,i-2,1,0);
            b[I] = Cb * (dy*aY[Im1y] + dx*aY[Im1x] + aY[Im2x]);
            aY[I] = C * ( -kappa*(dy*b[Im1y]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dy*aY[Im1y]+dx*aY[Im1x]) - (1+i-1)*(aY[Im2x]) );
            axY[Im1x] = aY[I]*i;
            ayY[Im1y] = aY[I];
            aL[I] = C * ( -(2*(1+i)-1)*(dy*aL[Im1y]+dx*aL[Im1x]) - (1+i-1)*(aL[Im2x]) );
            axL[Im1x] = aL[I]*i;
            ayL[Im1y] = aL[I];

            I    = getIndex(P,i,0,1);
            Im1z = I-1;
            Im1x = getIndex(P,i-1,0,1);
            Im2x = getIndex(P,i-2,0,1);
            b[I] = Cb * (dz*aY[Im1z] + dx*aY[Im1x] + aY[Im2x]);
            aY[I] = C * ( -kappa*(dz*b[Im1z]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dz*aY[Im1z]+dx*aY[Im1x]) - (1+i-1)*(aY[Im2x]) );
            axY[Im1x] = aY[I]*i;
            azY[Im1z] = aY[I];
            aL[I] = C * ( -(2*(1+i)-1)*(dz*aL[Im1z]+dx*aL[Im1x]) - (1+i-1)*(aL[Im2x]) );
            axL[Im1x] = aL[I]*i;
            azL[Im1z] = aL[I];

            I    = getIndex(P,0,i,1);
            Im1z = I-1;
            Im1y = I-(P+2-i);
            Im2y = Im1y-(P+2-i+1);
            b[I] = Cb * (dz*aY[Im1z] + dy*aY[Im1y] + aY[Im2y]);
            aY[I] = C * ( -kappa*(dz*b[Im1z]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dz*aY[Im1z]+dy*aY[Im1y]) - (1+i-1)*(aY[Im2y]) );
            ayY[Im1y] = aY[I]*i;
            azY[Im1z] = aY[I];
            aL[I] = C * ( -(2*(1+i)-1)*(dz*aL[Im1z]+dy*aL[Im1y]) - (1+i-1)*(aL[Im2y]) );
            ayL[Im1y] = aL[I]*i;
            azL[Im1z] = aL[I];
        }

        // One index 0, others >=2
        for (i=2; i<P+1; i++)
        {
            for (j=2; j<P+1-i; j++)
            {
                Cb   = -kappa/(i+j);
                C    = 1./((i+j)*R2);
                I    = getIndex(P,i,j,0);
                Im1x = getIndex(P,i-1,j,0);
                Im2x = getIndex(P,i-2,j,0);
                Im1y = I-(P+2-j-i);
                Im2y = Im1y-(P+3-j-i);
                b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + aY[Im2x] + aY[Im2y]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2x]+b[Im2y]) -(2*(i+j)-1)*(dx*aY[Im1x]+dy*aY[Im1y]) -(i+j-1)*(aY[Im2x]+aY[Im2y]) );
                axY[Im1x] = aY[I]*i;
                ayY[Im1y] = aY[I]*j;
                aL[I] = C * ( -(2*(i+j)-1)*(dx*aL[Im1x]+dy*aL[Im1y]) -(i+j-1)*(aL[Im2x]+aL[Im2y]) );
                axL[Im1x] = aL[I]*i;
                ayL[Im1y] = aL[I]*j;

                I    = getIndex(P,i,0,j);
                Im1x = getIndex(P,i-1,0,j);
                Im2x = getIndex(P,i-2,0,j);
                Im1z = I-1;
                Im2z = I-2;
                b[I] = Cb * (dx*aY[Im1x] + dz*aY[Im1z] + aY[Im2x] + aY[Im2z]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2x]+b[Im2z]) -(2*(i+j)-1)*(dx*aY[Im1x]+dz*aY[Im1z]) -(i+j-1)*(aY[Im2x]+aY[Im2z]) );
                axY[Im1x] = aY[I]*i;
                azY[Im1z] = aY[I]*j;
                aL[I] = C * ( -(2*(i+j)-1)*(dx*aL[Im1x]+dz*aL[Im1z]) -(i+j-1)*(aL[Im2x]+aL[Im2z]) );
                axL[Im1x] = aL[I]*i;
                azL[Im1z] = aL[I]*j;

                I    = getIndex(P,0,i,j);
                Im1y = I-(P+2-i);
                Im2y = Im1y-(P+3-i);
                Im1z = I-1;
                Im2z = I-2;
                b[I] = Cb * (dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2y] + aY[Im2z]);
                aY[I] = C * ( -kappa*(dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) -(2*(i+j)-1)*(dy*aY[Im1y]+dz*aY[Im1z]) -(i+j-1)*(aY[Im2y]+aY[Im2z]) );
                ayY[Im1y] = aY[I]*i;
                azY[Im1z] = aY[I]*j;
                aL[I] = C * ( -(2*(i+j)-1)*(dy*aL[Im1y]+dz*aL[Im1z]) -(i+j-1)*(aL[Im2y]+aL[Im2z]) );
                ayL[Im1y] = aL[I]*i;
                azL[Im1z] = aL[I]*j;
            }
        }

        if (P>2)
        {
            // Two index = 1, other>=1
            Cb   = -kappa/3;
            I    = getIndex(P,1,1,1);
            Im1x = getIndex(P,0,1,1);
            Im1y = getIndex(P,1,0,1);
            Im1y = I-(P);
            Im1z = I-1;
            b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z]);
            aY[I] = 1/(3*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]) -5*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) );
            axY[Im1x] = aY[I];
            ayY[Im1y] = aY[I];
            azY[Im1z] = aY[I];

            aL[I] = 1/(3*R2) * ( -5*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) );
            axL[Im1x] = aL[I];
            ayL[Im1y] = aL[I];
            azL[Im1z] = aL[I];

            for (i=2; i<P-1; i++)
            {
                Cb   = -kappa/(2+i);
                C    = 1./((i+2)*R2);
                I    = getIndex(P,i,1,1);
                Im1x = getIndex(P,i-1,1,1);
                Im1y = I-(P+2-i-1);
                Im1z = I-1;
                Im2x = getIndex(P,i-2,1,1);
                b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2x]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]) -(2*(i+2)-1)*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - (i+1)*(aY[Im2x]) );
                axY[Im1x] = aY[I]*i;
                ayY[Im1y] = aY[I];
                azY[Im1z] = aY[I];
                aL[I] = C * ( -(2*(i+2)-1)*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - (i+1)*(aL[Im2x]) );
                axL[Im1x] = aL[I]*i;
                ayL[Im1y] = aL[I];
                azL[Im1z] = aL[I];

                I    = getIndex(P,1,i,1);
                Im1x = getIndex(P,0,i,1);
                Im1y = I-(P+2-i-1);
                Im2y = Im1y-(P+3-i-1);
                Im1z = I-1 ;
                b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2y]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]) -(2*(i+2)-1)*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - (i+1)*(aY[Im2y]) );
                axY[Im1x] = aY[I];
                ayY[Im1y] = aY[I]*i;
                azY[Im1z] = aY[I];
                aL[I] = C * ( -(2*(i+2)-1)*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - (i+1)*(aL[Im2y]) );
                axL[Im1x] = aL[I];
                ayL[Im1y] = aL[I]*i;
                azL[Im1z] = aL[I];

                I    = getIndex(P,1,1,i);
                Im1x = getIndex(P,0,1,i);
                Im1y = I-(P);
                Im1z = I-1;
                Im2z = I-2;
                b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2z]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2z]) -(2*(i+2)-1)*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - (i+1)*(aY[Im2z]) );
                axY[Im1x] = aY[I];
                ayY[Im1y] = aY[I];
                azY[Im1z] = aY[I]*i;
                aL[I] = C * ( -(2*(i+2)-1)*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - (i+1)*(aL[Im2z]) );
                axL[Im1x] = aL[I];
                ayL[Im1y] = aL[I];
                azL[Im1z] = aL[I]*i;
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
                    I    = getIndex(P,1,i,j);
                    Im1x = getIndex(P,0,i,j);
                    Im1y = I-(P+2-1-i);
                    Im2y = Im1y-(P+3-1-i);
                    Im1z = I-1;
                    Im2z = I-2;
                    b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2y] + aY[Im2z]);
                    aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2y]+b[Im2z]) + C1*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - C2*(aY[Im2y]+aY[Im2z]) );
                    axY[Im1x] = aY[I];
                    ayY[Im1y] = aY[I]*i;
                    azY[Im1z] = aY[I]*j;
                    aL[I] = C * ( C1*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - C2*(aL[Im2y]+aL[Im2z]) );
                    axL[Im1x] = aL[I];
                    ayL[Im1y] = aL[I]*i;
                    azL[Im1z] = aL[I]*j;

                    I    = getIndex(P,i,1,j);
                    Im1x = getIndex(P,i-1,1,j);
                    Im1y = I-(P+2-i-1);
                    Im2x = getIndex(P,i-2,1,j);
                    Im1z = I-1;
                    Im2z = I-2;
                    b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2x] + aY[Im2z]);
                    aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2z]) + C1*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - C2*(aY[Im2x]+aY[Im2z]) );
                    axY[Im1x] = aY[I]*i;
                    ayY[Im1y] = aY[I];
                    azY[Im1z] = aY[I]*j;
                    aL[I] = C * ( C1*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - C2*(aL[Im2x]+aL[Im2z]) );
                    axL[Im1x] = aL[I]*i;
                    ayL[Im1y] = aL[I];
                    azL[Im1z] = aL[I]*j;

                    I    = getIndex(P,i,j,1);
                    Im1x = getIndex(P,i-1,j,1);
                    Im2x = getIndex(P,i-2,j,1);
                    Im1y = I-(P+2-i-j);
                    Im2y = Im1y-(P+3-i-j);
                    Im1z = I-1;
                    b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2x] + aY[Im2y]);
                    aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]) + C1*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - C2*(aY[Im2x]+aY[Im2y]) );
                    axY[Im1x] = aY[I]*i;
                    ayY[Im1y] = aY[I]*j;
                    azY[Im1z] = aY[I];
                    aL[I] = C * ( C1*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - C2*(aL[Im2x]+aL[Im2y]) );
                    axL[Im1x] = aL[I]*i;
                    ayL[Im1y] = aL[I]*j;
                    azL[Im1z] = aL[I];
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
                        I    = getIndex(P,i,j,k);
                        Im1x = getIndex(P,i-1,j,k);
                        Im2x = getIndex(P,i-2,j,k);
                        Im1y = I-(P+2-i-j);
                        Im2y = Im1y-(P+3-i-j);
                        Im1z = I-1;
                        Im2z = I-2;
                        b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2x] + aY[Im2y] + aY[Im2z]);
                        aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]+b[Im2y]+b[Im2z]) + C1*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - C2*(aY[Im2x]+aY[Im2y]+aY[Im2z]) );
                        axY[Im1x] = aY[I]*i;
                        ayY[Im1y] = aY[I]*j;
                        azY[Im1z] = aY[I]*k;

                        aL[I] = C * ( C1*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - C2*(aL[Im2x]+aL[Im2y]+aL[Im2z]) );
                        axL[Im1x] = aL[I]*i;
                        ayL[Im1y] = aL[I]*j;
                        azL[Im1z] = aL[I]*k;
                    }
                }
            }
        }


    }

	// Multiply coefficients  with corresponding multipoles
    __device__ void multipole(REAL &L, REAL &dL, REAL &Y, REAL &dY, 
                            REAL *mp, REAL *mpx, REAL *mpy, REAL *mpz, 
                            REAL *aL, REAL *axL, REAL *ayL, REAL *azL, 
                            REAL *aY, REAL *axY, REAL *ayY, REAL *azY, 
                            int CJ_start, int jblock, int j, REAL E_hat)
    {
        int offset;
        for (int i=0; i<Nm; i++)
        {
            offset = (CJ_start+j)*Nm + jblock*BSZ*Nm + i;
            L  += mp[offset] * aL[i];
            Y  += E_hat * mp[offset] * aY[i];
            dL += mpx[offset]*axL[i] + mpy[offset]*ayL[i] + mpz[offset]*azL[i];
            dY += mpx[offset]*axY[i] + mpy[offset]*ayY[i] + mpz[offset]*azY[i];
        }

    }

	// norm of a vector
    __device__ REAL norm(REAL *x)
    {
        return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    }

	// cross product between two vectors
    __device__ void cross(REAL *x, REAL *y, REAL *z) // z is the resulting array
    {
        z[0] = x[1]*y[2] - x[2]*y[1];
        z[1] = x[2]*y[0] - x[0]*y[2];
        z[2] = x[0]*y[1] - x[1]*y[0];
    }

	// 3x3 matvec
    __device__ void MV(REAL *M, REAL *V, REAL *res) // 3x3 mat-vec
    {
        REAL V2[3] = {V[0], V[1], V[2]};
        for (int i=0; i<3; i++)
        {
            REAL sum = 0.;
            for (int j=0; j<3; j++)
            {
                sum += M[3*i+j]*V2[j];
            }
            res[i] = sum;
        }
    }

	 // len(3) vector dot product 
    __device__ REAL dot_prod(REAL *x, REAL *y)
    {
        return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
    }

    __device__ void axpy(REAL *x, REAL *y, REAL *z, REAL alpha, int sign, int N)
    {
        for(int i=0; i<N; i++)
        {
            z[i] = sign*alpha*x[i] + y[i];
        }
    }

    __device__ void ax(REAL *x, REAL *y, REAL alpha, int N)
    {
        for(int i=0; i<N; i++)
        {
            y[i] = alpha*x[i];
        }

    }

	// integral over line
    __device__ void lineInt(REAL *PHI, REAL z, REAL x, REAL v1, REAL v2, REAL kappa, REAL *xk, REAL *wk, int K)
    {
        REAL theta1 = atan2(v1,x);
        REAL theta2 = atan2(v2,x);
        REAL dtheta = theta2 - theta1;
        REAL thetam = (theta2 + theta1)/2;


        REAL absZ = fabs(z), signZ;
        if (absZ<1e-10) signZ = 0;
        else            signZ = z/absZ;

        // Loop over gauss points
        REAL thetak, Rtheta, R, expKr, expKz = exp(-kappa*absZ);
        for (int i=0; i<K; i++)
        {
            thetak = dtheta/2*xk[i] + thetam;
            Rtheta = x/cos(thetak);
            R      = sqrt(Rtheta*Rtheta + z*z);
            expKr  = exp(-kappa*R);
            PHI[0]+= -wk[i]*(expKr - expKz)/kappa * dtheta/2;
            PHI[1]+= -wk[i]*(z/R*expKr - expKz*signZ) * dtheta/2;
            PHI[2]+=  wk[i]*(R-absZ) * dtheta/2;
            PHI[3]+= -wk[i]*(z/R - signZ) * dtheta/2;
        }
    }

    __device__ void intSide(REAL *PHI, REAL *v1, REAL *v2, REAL p, REAL kappa, REAL *xk, REAL *wk, int K)
    {
        REAL v21[3];
        for (int i=0; i<3; i++)
        {
            v21[i] = v2[i] - v1[i];
        }

        REAL L21 = norm(v21);
        REAL v21u[3];
        ax(v21, v21u, 1/L21, 3);

        REAL unit[3] = {0.,0.,1.};
        REAL orthog[3];
        cross(unit, v21u, orthog);

        REAL alpha = dot_prod(v21,v1)/(L21*L21);

        REAL rOrthog[3];
        axpy(v21, v1, rOrthog, alpha, -1, 3);

        REAL d_toEdge = norm(rOrthog);
        REAL v1_neg[3];
        ax(v1, v1_neg, -1, 3);

        REAL side_vec[3];
        cross(v21, v1_neg, side_vec);

        REAL rotateToVertLine[9];

        for(int i=0; i<3; i++)
        {
            rotateToVertLine[3*i] = orthog[i];
            rotateToVertLine[3*i+1] = v21u[i];
            rotateToVertLine[3*i+2] = unit[i];
        }

        REAL v1new[3];
        MV(rotateToVertLine,v1,v1new);

        if (v1new[0]<0)
        {
            ax(v21u, v21u, -1, 3);
            ax(orthog, orthog, -1, 3);
            ax(rotateToVertLine, rotateToVertLine, -1, 9);
            rotateToVertLine[8] = 1.;
            MV(rotateToVertLine,v1,v1new);
        }

        REAL v2new[3], rOrthognew[3];
        MV(rotateToVertLine,v2,v2new);
        MV(rotateToVertLine,rOrthog,rOrthognew);
        REAL x = v1new[0];

        if ((v1new[1]>0 && v2new[1]<0) || (v1new[1]<0 && v2new[1]>0))
        {
            REAL PHI1[4] = {0.,0.,0.,0.} , PHI2[4] = {0.,0.,0.,0.};
            lineInt(PHI1, p, x, 0, v1new[1], kappa, xk, wk, K);
            lineInt(PHI2, p, x, v2new[1], 0, kappa, xk, wk, K);

            for(int i=0; i<4; i++)
                PHI[i] += PHI1[i] + PHI2[i];
        }
        else
        {
            REAL PHI_aux[4] = {0.,0.,0.,0.};
            lineInt(PHI_aux, p, x, v1new[1], v2new[1], kappa, xk, wk, K);

            for(int i=0; i<4; i++)
                PHI[i] -= PHI_aux[i];
        }

    }

	// Semi-analytical integral
    __device__ void SA(REAL *PHI, REAL *y, REAL x0, REAL x1, REAL x2, 
                        REAL kappa, int same, REAL *xk, REAL *wk, int K)
    {   
        REAL y0_panel[3], y1_panel[3], y2_panel[3], x_panel[3];
        REAL X[3], Y[3], Z[3];

        x_panel[0] = x0 - y[0];
        x_panel[1] = x1 - y[1];
        x_panel[2] = x2 - y[2];
        for (int i=0; i<3; i++)
        {
            y0_panel[i] = 0.;
            y1_panel[i] = y[3+i] - y[i];
            y2_panel[i] = y[6+i] - y[i];
            X[i] = y1_panel[i];
        }


        // Find panel coordinate system X: 0->1
        cross(y1_panel, y2_panel, Z); 
        REAL Xnorm = norm(X); 
        REAL Znorm = norm(Z); 
        for (int i=0; i<3; i++)
        {   
            X[i] /= Xnorm;
            Z[i] /= Znorm;
        }   

        cross(Z,X,Y);

        // Rotate the coordinate system to match panel plane
        REAL rot_matrix[9];
        for (int i=0; i<3; i++)
        {   
            rot_matrix[i] = X[i];
            rot_matrix[i+3] = Y[i];
            rot_matrix[i+6] = Z[i];
        }   
        
        REAL panel0_plane[3], panel1_plane[3], panel2_plane[3], x_plane[3];
        MV(rot_matrix, y0_panel, panel0_plane);
        MV(rot_matrix, y1_panel, panel1_plane);
        MV(rot_matrix, y2_panel, panel2_plane);
        MV(rot_matrix, x_panel, x_plane);

        // Shift origin so it matches collocation point
        REAL panel0_final[3], panel1_final[3], panel2_final[3];
        for (int i=0; i<3; i++)
        {   
            if (i<2)
            {   
                panel0_final[i] = panel0_plane[i] - x_plane[i]; 
                panel1_final[i] = panel1_plane[i] - x_plane[i]; 
                panel2_final[i] = panel2_plane[i] - x_plane[i]; 
            }   
            else
            {   
                panel0_final[i] = panel0_plane[i]; 
                panel1_final[i] = panel1_plane[i]; 
                panel2_final[i] = panel2_plane[i]; 
            }   
        }   

        // Loop over sides
        intSide(PHI, panel0_final, panel1_final, x_plane[2], kappa, xk, wk, K); // Side 0
        intSide(PHI, panel1_final, panel2_final, x_plane[2], kappa, xk, wk, K); // Side 1
        intSide(PHI, panel2_final, panel0_final, x_plane[2], kappa, xk, wk, K); // Side 2

        if (same==1)
        {
            PHI[1] = 2*M_PI;
            PHI[3] = -2*M_PI;
        }
    }

	// M2P 
    __global__ void M2P(int *sizeTarDev, int *offsetMltDev,
                        REAL *xtDev, REAL *ytDev, REAL *ztDev,
                        REAL *xcDev, REAL *ycDev, REAL *zcDev, 
                        REAL *mpDev, REAL *mpxDev, REAL *mpyDev, REAL *mpzDev,
                        REAL *Pre0, REAL *Pre1, REAL *Pre2, REAL *Pre3,
                        REAL *p1, REAL *p2, int N, int P, REAL kappa, REAL E_hat, int BpT, int NCRIT)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int CJ_start = offsetMltDev[blockIdx.x];
        int Nmlt     = offsetMltDev[blockIdx.x+1] - CJ_start;

        REAL xi, yi, zi,
             dx, dy, dz; 
        REAL aL[Nm], axL[Nm], ayL[Nm], azL[Nm],
             aY[Nm], axY[Nm], ayY[Nm], azY[Nm];

        __shared__ REAL xc_sh[BSZ],
                        yc_sh[BSZ],
                        zc_sh[BSZ];
        int i;

		// initialize coefficient arrays with zeros
        for (int mult=0; mult<Nm; mult++)
        {
            axL[mult] = 0.;
            ayL[mult] = 0.;
            azL[mult] = 0.;
            axY[mult] = 0.;
            ayY[mult] = 0.;
            azY[mult] = 0.;
        }

		// One thread does NCRIT/BSZ rows (targets)
        for (int iblock=0; iblock<BpT; iblock++)
        {
            i  = I + iblock*BSZ;
            xi = xtDev[i];
            yi = ytDev[i];
            zi = ztDev[i];
            
            REAL L = 0., dL = 0., Y = 0., dY = 0.;

            for(int jblock=0; jblock<(Nmlt-1)/BSZ; jblock++)
            {
				// Load to shared memory
                __syncthreads();
                xc_sh[threadIdx.x] = xcDev[CJ_start + jblock*BSZ + threadIdx.x];
                yc_sh[threadIdx.x] = ycDev[CJ_start + jblock*BSZ + threadIdx.x];
                zc_sh[threadIdx.x] = zcDev[CJ_start + jblock*BSZ + threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
                {
                    for (int j=0; j<BSZ; j++)
                    {
                        dx = xi - xc_sh[j];
                        dy = yi - yc_sh[j];
                        dz = zi - zc_sh[j];
						// get Taylor coefficients
                        getCoeff(aL, axL, ayL, azL, aY, axY, ayY, azY, 
                                dx, dy, dz, kappa, P);
						// Multiply Taylor coefficients with corresponding multipole
                        multipole(L, dL, Y, dY, mpDev, mpxDev, mpyDev, mpzDev, 
                                aL, axL, ayL, azL, aY, axY, ayY, azY, 
                                CJ_start, jblock, j, E_hat);
                    }
                }
            } 

			// If N is not a multiple of block size
            __syncthreads();
			// Load to shared memory
            int jblock = (Nmlt-1)/BSZ;
            xc_sh[threadIdx.x] = xcDev[CJ_start + jblock*BSZ + threadIdx.x];
            yc_sh[threadIdx.x] = ycDev[CJ_start + jblock*BSZ + threadIdx.x];
            zc_sh[threadIdx.x] = zcDev[CJ_start + jblock*BSZ + threadIdx.x];
            __syncthreads();
            
            if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
            {
                for (int j=0; j<Nmlt-(jblock*BSZ); j++)
                {
                    dx = xi - xc_sh[j];
                    dy = yi - yc_sh[j];
                    dz = zi - zc_sh[j];
                    getCoeff(aL, axL, ayL, azL, aY, axY, ayY, azY, 
                            dx, dy, dz, kappa, P);
                    multipole(L, dL, Y, dY, mpDev, mpxDev, mpyDev, mpzDev, 
                            aL, axL, ayL, azL, aY, axY, ayY, azY, 
                            CJ_start, jblock, j, E_hat);
                }
            }

			// Add contribution
            if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
            {
                // With preconditioner
                //p1[i] += Pre0[i]*(dL+L) - Pre1[i]*(dY+Y);
                //p2[i] += Pre2[i]*(dL+L) - Pre3[i]*(dY+Y);
                // No preconditioner
                p1[i] += dL + L;
                p2[i] += -dY - Y;
            }
        }
        
    }


	// P2P
    __global__ void P2P(int *offsetSrcDev, int *sizeTarDev, int *tri, int *kDev, 
                        REAL *xsDev, REAL *ysDev, REAL *zsDev, REAL *mDev, REAL *mxDev, 
                        REAL *myDev, REAL *mzDev, REAL *xtDev, REAL *ytDev, REAL *ztDev, 
                        REAL *AreaDev, REAL *p1, REAL *p2,REAL *Pre0, REAL *Pre1, REAL *Pre2, 
                        REAL *Pre3, REAL E_hat, int N, REAL *vertexDev, REAL *normal_xDev, 
                        int K, REAL w0, REAL *xkDev, REAL *wkDev, REAL kappa, 
                        REAL threshold, REAL eps, int BpT, int NCRIT, int *AI_int_gpu)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int CJ_start = offsetSrcDev[blockIdx.x];
        int Nsrc     = offsetSrcDev[blockIdx.x+1] - CJ_start;
        
        REAL xi, yi, zi, dx, dy, dz, r, R_tri, dx_tri, dy_tri, dz_tri, L_d;

        __shared__ REAL ver_sh[9*BSZ], xc_sh[BSZ], yc_sh[BSZ], zc_sh[BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], nx_sh[BSZ],
                        xk_sh[xkSize], wk_sh[xkSize];


        if (threadIdx.x<xkSize)
        {
            xk_sh[threadIdx.x] = xkDev[threadIdx.x];
            wk_sh[threadIdx.x] = wkDev[threadIdx.x];
        }
        __syncthreads();

        int i, same;

        for (int iblock=0; iblock<BpT; iblock++)
        {
            REAL L = 0., Y = 0., dL = 0., dY = 0.;
            i  = I + iblock*BSZ;
            xi = xtDev[i];
            yi = ytDev[i];
            zi = ztDev[i];
            int an_counter = 0;

            for(int jblock=0; jblock<(Nsrc-1)/BSZ; jblock++)
            {
				// Load to shared memory
                __syncthreads();
                xj_sh[threadIdx.x] = xsDev[CJ_start + jblock*BSZ + threadIdx.x];
                yj_sh[threadIdx.x] = ysDev[CJ_start + jblock*BSZ + threadIdx.x];
                zj_sh[threadIdx.x] = zsDev[CJ_start + jblock*BSZ + threadIdx.x];
                m_sh[threadIdx.x] = mDev[CJ_start + jblock*BSZ + threadIdx.x];
                mx_sh[threadIdx.x] = mxDev[CJ_start + jblock*BSZ + threadIdx.x];
                my_sh[threadIdx.x] = myDev[CJ_start + jblock*BSZ + threadIdx.x];
                mz_sh[threadIdx.x] = mzDev[CJ_start + jblock*BSZ + threadIdx.x];
                A_sh[threadIdx.x] = AreaDev[CJ_start + jblock*BSZ + threadIdx.x];
                k_sh[threadIdx.x] = kDev[CJ_start + jblock*BSZ + threadIdx.x];
                nx_sh[threadIdx.x] = normal_xDev[CJ_start + jblock*BSZ + threadIdx.x];

                for (int vert=0; vert<9; vert++)
                {
                    ver_sh[9*threadIdx.x+vert] = vertexDev[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                }
                __syncthreads();

				// Calculate center of mass of triangle
                xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
                yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
                zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
                {
					// Loop over sources
                    for (int j=0; j<BSZ; j++)
                    {
                        dx_tri = xi - xc_sh[j];
                        dy_tri = yi - yc_sh[j];
                        dz_tri = zi - zc_sh[j];
                        R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri + eps);

                        L_d = sqrt(2*A_sh[j])/(R_tri);

						// if far: Gauss quadrature
                        if(L_d<threshold)
                        {
                            dx = xj_sh[j] - xi;
                            dy = yj_sh[j] - yi;
                            dz = zj_sh[j] - zi;
                            r = sqrt(dx*dx + dy*dy + dz*dz + eps);
                            L  += m_sh[j]/r; 
                            Y  += m_sh[j] * exp(-kappa*r)/r; 
                            dY += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*exp(-kappa*r)/(r*r)*(kappa+1/r); 
                            dL += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                        }
						// if close: semi-analytical integration
                        else if(k_sh[j]==0)
                        {
                            REAL PHI_1[4] = {0., 0., 0., 0.};
                            REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                            ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                            ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                            // Check if it is same triangle
                            REAL x_av=0., y_av=0., z_av=0.;
                            for (int kk=0; kk<K; kk++)
                            {
                                x_av += xtDev[i+kk];
                                y_av += ytDev[i+kk];
                                z_av += ztDev[i+kk];
                            }

                            x_av /= K; // Average of Gauss points is triangle centroid
                            y_av /= K;
                            z_av /= K;

                            dx = x_av - xc_sh[j]; 
                            dy = y_av - yc_sh[j]; 
                            dz = z_av - zc_sh[j]; 
                            r  = sqrt(dx*dx + dy*dy + dz*dz);
                            if (r<1e-8) same=1;
                            else        same=0;
                            

                            SA(PHI_1, panel, xi, yi, zi, kappa, same, xk_sh, wk_sh, xkSize);
                
                            Y  += PHI_1[0] * m_sh[j]/(w0*A_sh[j]);
                            dY += PHI_1[1] * mx_sh[j]/(w0*A_sh[j]*nx_sh[j]);
                            L  += PHI_1[2] * m_sh[j]/(w0*A_sh[j]);
                            dL += PHI_1[3] * mx_sh[j]/(w0*A_sh[j]*nx_sh[j]);
                            an_counter += 1;
                        }
                    }
                }
            }
           
			// If N is not a multiple of Block size
            __syncthreads();
			// Load to shared memory
            int jblock = (Nsrc-1)/BSZ;
            xj_sh[threadIdx.x] = xsDev[CJ_start + jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = ysDev[CJ_start + jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zsDev[CJ_start + jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x] = mDev[CJ_start + jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mxDev[CJ_start + jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = myDev[CJ_start + jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mzDev[CJ_start + jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x] = AreaDev[CJ_start + jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x] = kDev[CJ_start + jblock*BSZ + threadIdx.x];
            nx_sh[threadIdx.x] = normal_xDev[CJ_start + jblock*BSZ + threadIdx.x];

            for (int vert=0; vert<9; vert++)
            {
                ver_sh[9*threadIdx.x+vert] = vertexDev[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
            }
            __syncthreads();

            xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
            yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
            zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
            __syncthreads();
			// Loop over sources
            if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
            {
                for (int j=0; j<Nsrc-(jblock*BSZ); j++)
                {
                    dx_tri = xi - xc_sh[j];
                    dy_tri = yi - yc_sh[j];
                    dz_tri = zi - zc_sh[j];
                    R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri + eps);

                    L_d = sqrt(2*A_sh[j])/(R_tri);

					// If far: gauss quadrature
                    if (L_d<threshold)
                    {
                        dx = xj_sh[j] - xi;
                        dy = yj_sh[j] - yi;
                        dz = zj_sh[j] - zi;
                        r = sqrt(dx*dx + dy*dy + dz*dz + eps);
                        L  += m_sh[j]/r; 
                        Y  += m_sh[j] * exp(-kappa*r)/r; 
                        dY += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*exp(-kappa*r)/(r*r)*(kappa+1/r); 
                        dL += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                    }

					// If close: semi-analytical integration
                    else if(k_sh[j]==0)
                    {
                        REAL PHI_1[4] = {0., 0., 0., 0.};
                        REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                        ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                        ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                        // Check if it is same triangle
                        REAL x_av=0., y_av=0., z_av=0.;
                        for (int kk=0; kk<K; kk++)
                        {
                            x_av += xtDev[i+kk];
                            y_av += ytDev[i+kk];
                            z_av += ztDev[i+kk];
                        }

                        x_av /= K; // Average of Gauss points is triangle centroid
                        y_av /= K;
                        z_av /= K;

                        dx = x_av - xc_sh[j]; 
                        dy = y_av - yc_sh[j]; 
                        dz = z_av - zc_sh[j]; 
                        r  = sqrt(dx*dx + dy*dy + dz*dz);
                        if (r<1e-8) same=1;
                        else        same=0;

                        SA(PHI_1, panel, xi, yi, zi, kappa, same, xk_sh, wk_sh, xkSize);
            
                        Y  += PHI_1[0] * m_sh[j]/(w0*A_sh[j]);
                        dY += PHI_1[1] * mx_sh[j]/(w0*A_sh[j]*nx_sh[j]);
                        L  += PHI_1[2] * m_sh[j]/(w0*A_sh[j]);
                        dL += PHI_1[3] * mx_sh[j]/(w0*A_sh[j]*nx_sh[j]);
                        an_counter += 1;
                    }
                }
            }
        
            if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
            {
                // With preconditioner
                //p1[i] += Pre0[i]*(dL+L) - Pre1[i]*(dY+E_hat*Y);
                //p2[i] += Pre2[i]*(dL+L) - Pre3[i]*(dY+E_hat*Y);
                // No preconditioner
                p1[i] += dL + L;
                p2[i] += -dY - E_hat*Y;
                AI_int_gpu[i] = an_counter;
            }
        }
    }


    """%{'blocksize':BSZ, 'Nmult':Nm, 'K_1D':xkSize}, nvcc="nvcc")#, options=["-Xptxas=-v"])

    return mod
