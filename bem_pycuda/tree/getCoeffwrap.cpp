/**
This code is a modification of TREECODE3D_YUKAWA from Hans Johnston in UMass Amherst.
We thank them for opening and aloow use of their code
**/
#include <cmath>
#include <stdio.h>
#include <iostream>
#define REAL double

int getIndex(int P, int i, int j, int k)
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
        indices[iter] = getIndex(P, ii[iter], jj[iter], kk[iter]);

}

void getCoeff(REAL *aY, int aYSize, REAL *axY, int axYSize, REAL *ayY, int ayYSize, 
              REAL *azY, int azYSize, REAL *aL, int aLSize, REAL *axL, int axLSize, 
              REAL *ayL, int ayLSize, REAL *azL, int azLSize, 
              REAL dx, REAL dy, REAL dz, int P, REAL kappa)
{
    REAL b[aYSize];
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

void multipole(REAL *L , int LSize, 
               REAL *dL, int dLSize, 
               REAL *Y , int YSize, 
               REAL *dY, int dYSize, 
               REAL *M , int MSize, 
               REAL *Mx, int MxSize, 
               REAL *My, int MySize, 
               REAL *Mz, int MzSize, 
               REAL *dx, int dxSize, 
               REAL *dy, int dySize, 
               REAL *dz, int dzSize,
              int P, REAL kappa, int Nm, REAL E_hat)
{
    REAL aL[Nm], axL[Nm], ayL[Nm], azL[Nm];
    REAL aY[Nm], axY[Nm], ayY[Nm], azY[Nm];

    for (int i=0; i<LSize; i++)
    {   
        for (int ii=0; ii<Nm; ii++)
        {   
            aL[ii] = 0.; 
            axL[ii] = 0.; 
            ayL[ii] = 0.; 
            azL[ii] = 0.; 
            aY[ii] = 0.; 
            axY[ii] = 0.; 
            ayY[ii] = 0.; 
            azY[ii] = 0.; 
        }   

        getCoeff(aY, Nm, axY, Nm, ayY, Nm, azY, Nm, 
                 aL, Nm, axL, Nm, ayL, Nm, azL, Nm, 
                 dx[i], dy[i], dz[i], P, kappa);

        for (int j=0; j<Nm; j++)
        {   
            L[i] += aL[j]*M[j];
            Y[i] += E_hat*aY[j]*M[j];
            dL[i] += axL[j]*Mx[j] + ayL[j]*My[j] + azL[j]*Mz[j];
            dY[i] += axY[j]*Mx[j] + ayY[j]*My[j] + azY[j]*Mz[j];
        }   

    }   
}

/*
int main()
{
    REAL dx=0.1, dy=0.2, dz=0.1;
    int P=3;
    REAL *aY = new REAL[20];

    getCoeff(aY, dx, dy, dz, P); 

    for (int i=0; i<20; i++)
        printf ("%f\t",aY[i]);
    printf("\n");
    return 0;
}
*/
