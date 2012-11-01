from pycuda.compiler import SourceModule

def kernels(BSZ, Nm, xkSize, P):
    
    mod = SourceModule( """

    #define REAL double
    #define BSZ %(blocksize)d
    #define Nm  %(Nmult)d
    #define xkSize %(K_1D)d
    #define P      %(Ptree)d


    __device__ int getIndex(int i, int j, int k, int *Index)
    {   
        return Index[(P+1)*(P+1)*i + (P+1)*j + k]; 
    }


    __device__ void getCoeff(REAL *aL, REAL *axL, REAL *ayL, REAL *azL, 
                             REAL *aY, REAL *axY, REAL *ayY, REAL *azY,
                             REAL dx, REAL dy, REAL dz, REAL kappa, int *Index)
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
        I = getIndex(1,0,0,Index);
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
            I    = getIndex(i,0,0,Index);
            Im1x = getIndex(i-1,0,0,Index);
            Im2x = getIndex(i-2,0,0,Index);
            b[I] = Cb * (dx*aY[Im1x] + aY[Im2x]);
            aY[I] = C * ( -kappa*(dx*b[Im1x] + b[Im2x]) -(2*i-1)*dx*aY[Im1x] - (i-1)*aY[Im2x] );
            axY[Im1x] = aY[I]*i;
            aL[I] = C * ( -(2*i-1)*dx*aL[Im1x] - (i-1)*aL[Im2x] );
            axL[Im1x] = aL[I]*i;


            I    = getIndex(0,i,0,Index);
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
        I    = getIndex(1,1,0,Index);
        Im1x = P+1;
        Im1y = I-(P+2-1-1);
        b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y]);
        aY[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]) -(2*2-1)*(dx*aY[Im1x]+dy*aY[Im1y]) );
        axY[Im1x] = aY[I];
        ayY[Im1y] = aY[I];
        aL[I] = 1./(2*R2) * ( -(2*2-1)*(dx*aL[Im1x]+dy*aL[Im1y]) );
        axL[Im1x] = aL[I];
        ayL[Im1y] = aL[I];

        I    = getIndex(1,0,1,Index);
        Im1x = 1;
        Im1z = I-1;
        b[I] = Cb * (dx*aY[Im1x] + dz*aY[Im1z]);
        aY[I] = 1./(2*R2) * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]) -(2*2-1)*(dx*aY[Im1x]+dz*aY[Im1z]) );
        axY[Im1x] = aY[I];
        azY[Im1z] = aY[I];
        aL[I] = 1./(2*R2) * ( -(2*2-1)*(dx*aL[Im1x]+dz*aL[Im1z]) );
        axL[Im1x] = aL[I];
        azL[Im1z] = aL[I];

        I    = getIndex(0,1,1,Index);
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
            I    = getIndex(1,i,0,Index);
            Im1x = getIndex(0,i,0,Index);
            Im1y = I-(P+2-i-1);
            Im2y = Im1y-(P+2-i);
            b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + aY[Im2y]);
            aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2y]) -(2*(1+i)-1)*(dx*aY[Im1x]+dy*aY[Im1y]) - (1+i-1)*(aY[Im2y]) );
            axY[Im1x] = aY[I];
            ayY[Im1y] = aY[I]*i;
            aL[I] = C * ( -(2*(1+i)-1)*(dx*aL[Im1x]+dy*aL[Im1y]) - (1+i-1)*(aL[Im2y]) );
            axL[Im1x] = aL[I];
            ayL[Im1y] = aL[I]*i;

            I    = getIndex(1,0,i,Index);
            Im1x = getIndex(0,0,i,Index);
            Im1z = I-1;
            Im2z = I-2;
            b[I] = Cb * (dx*aY[Im1x] + dz*aY[Im1z] + aY[Im2z]);
            aY[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2z]) -(2*(1+i)-1)*(dx*aY[Im1x]+dz*aY[Im1z]) - (1+i-1)*(aY[Im2z]) );
            axY[Im1x] = aY[I];
            azY[Im1z] = aY[I]*i;
            aL[I] = C * ( -(2*(1+i)-1)*(dx*aL[Im1x]+dz*aL[Im1z]) - (1+i-1)*(aL[Im2z]) );
            axL[Im1x] = aL[I];
            azL[Im1z] = aL[I]*i;

            I    = getIndex(0,1,i,Index);
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

            I    = getIndex(i,1,0,Index);
            Im1y = I-(P+2-1-i);
            Im1x = getIndex(i-1,1,0,Index);
            Im2x = getIndex(i-2,1,0,Index);
            b[I] = Cb * (dy*aY[Im1y] + dx*aY[Im1x] + aY[Im2x]);
            aY[I] = C * ( -kappa*(dy*b[Im1y]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dy*aY[Im1y]+dx*aY[Im1x]) - (1+i-1)*(aY[Im2x]) );
            axY[Im1x] = aY[I]*i;
            ayY[Im1y] = aY[I];
            aL[I] = C * ( -(2*(1+i)-1)*(dy*aL[Im1y]+dx*aL[Im1x]) - (1+i-1)*(aL[Im2x]) );
            axL[Im1x] = aL[I]*i;
            ayL[Im1y] = aL[I];

            I    = getIndex(i,0,1,Index);
            Im1z = I-1;
            Im1x = getIndex(i-1,0,1,Index);
            Im2x = getIndex(i-2,0,1,Index);
            b[I] = Cb * (dz*aY[Im1z] + dx*aY[Im1x] + aY[Im2x]);
            aY[I] = C * ( -kappa*(dz*b[Im1z]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dz*aY[Im1z]+dx*aY[Im1x]) - (1+i-1)*(aY[Im2x]) );
            axY[Im1x] = aY[I]*i;
            azY[Im1z] = aY[I];
            aL[I] = C * ( -(2*(1+i)-1)*(dz*aL[Im1z]+dx*aL[Im1x]) - (1+i-1)*(aL[Im2x]) );
            axL[Im1x] = aL[I]*i;
            azL[Im1z] = aL[I];

            I    = getIndex(0,i,1,Index);
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
                I    = getIndex(i,j,0,Index);
                Im1x = getIndex(i-1,j,0,Index);
                Im2x = getIndex(i-2,j,0,Index);
                Im1y = I-(P+2-j-i);
                Im2y = Im1y-(P+3-j-i);
                b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + aY[Im2x] + aY[Im2y]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+b[Im2x]+b[Im2y]) -(2*(i+j)-1)*(dx*aY[Im1x]+dy*aY[Im1y]) -(i+j-1)*(aY[Im2x]+aY[Im2y]) );
                axY[Im1x] = aY[I]*i;
                ayY[Im1y] = aY[I]*j;
                aL[I] = C * ( -(2*(i+j)-1)*(dx*aL[Im1x]+dy*aL[Im1y]) -(i+j-1)*(aL[Im2x]+aL[Im2y]) );
                axL[Im1x] = aL[I]*i;
                ayL[Im1y] = aL[I]*j;

                I    = getIndex(i,0,j,Index);
                Im1x = getIndex(i-1,0,j,Index);
                Im2x = getIndex(i-2,0,j,Index);
                Im1z = I-1;
                Im2z = I-2;
                b[I] = Cb * (dx*aY[Im1x] + dz*aY[Im1z] + aY[Im2x] + aY[Im2z]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dz*b[Im1z]+b[Im2x]+b[Im2z]) -(2*(i+j)-1)*(dx*aY[Im1x]+dz*aY[Im1z]) -(i+j-1)*(aY[Im2x]+aY[Im2z]) );
                axY[Im1x] = aY[I]*i;
                azY[Im1z] = aY[I]*j;
                aL[I] = C * ( -(2*(i+j)-1)*(dx*aL[Im1x]+dz*aL[Im1z]) -(i+j-1)*(aL[Im2x]+aL[Im2z]) );
                axL[Im1x] = aL[I]*i;
                azL[Im1z] = aL[I]*j;

                I    = getIndex(0,i,j,Index);
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
            I    = getIndex(1,1,1,Index);
            Im1x = getIndex(0,1,1,Index);
            Im1y = getIndex(1,0,1,Index);
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
                I    = getIndex(i,1,1,Index);
                Im1x = getIndex(i-1,1,1,Index);
                Im1y = I-(P+2-i-1);
                Im1z = I-1;
                Im2x = getIndex(i-2,1,1,Index);
                b[I] = Cb * (dx*aY[Im1x] + dy*aY[Im1y] + dz*aY[Im1z] + aY[Im2x]);
                aY[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]) -(2*(i+2)-1)*(dx*aY[Im1x]+dy*aY[Im1y]+dz*aY[Im1z]) - (i+1)*(aY[Im2x]) );
                axY[Im1x] = aY[I]*i;
                ayY[Im1y] = aY[I];
                azY[Im1z] = aY[I];
                aL[I] = C * ( -(2*(i+2)-1)*(dx*aL[Im1x]+dy*aL[Im1y]+dz*aL[Im1z]) - (i+1)*(aL[Im2x]) );
                axL[Im1x] = aL[I]*i;
                ayL[Im1y] = aL[I];
                azL[Im1z] = aL[I];

                I    = getIndex(1,i,1,Index);
                Im1x = getIndex(0,i,1,Index);
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

                I    = getIndex(1,1,i,Index);
                Im1x = getIndex(0,1,i,Index);
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
                    I    = getIndex(1,i,j,Index);
                    Im1x = getIndex(0,i,j,Index);
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

                    I    = getIndex(i,1,j,Index);
                    Im1x = getIndex(i-1,1,j,Index);
                    Im1y = I-(P+2-i-1);
                    Im2x = getIndex(i-2,1,j,Index);
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

                    I    = getIndex(i,j,1,Index);
                    Im1x = getIndex(i-1,j,1,Index);
                    Im2x = getIndex(i-2,j,1,Index);
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
                        I    = getIndex(i,j,k,Index);
                        Im1x = getIndex(i-1,j,k,Index);
                        Im2x = getIndex(i-2,j,k,Index);
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

    __device__ REAL norm(REAL *x)
    {
        return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    }

    __device__ void cross(REAL *x, REAL *y, REAL *z) // z is the resulting array
    {
        z[0] = x[1]*y[2] - x[2]*y[1];
        z[1] = x[2]*y[0] - x[0]*y[2];
        z[2] = x[0]*y[1] - x[1]*y[0];
    }

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

    __device__ void MVip(REAL *M, REAL *V) // 3x3 mat-vec in-place
    {
        REAL V2[3] = {V[0], V[1], V[2]};
        for (int i=0; i<3; i++)
        {
            REAL sum = 0.;
            for (int j=0; j<3; j++)
            {
                sum += M[3*i+j]*V2[j];
            }
            V[i] = sum;
        }
    }

    __device__ REAL dot_prod(REAL *x, REAL *y) // len(3) vector dot product
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

    __device__ void axip(REAL *x, REAL alpha, int N)
    {
        for(int i=0; i<N; i++)
        {
            x[i] = alpha*x[i];
        }

    }

    __device__ void lineInt(REAL *PHI, REAL z, REAL x, REAL v1, REAL v2, REAL kappa, REAL *xk, REAL *wk, int K)
    {
        REAL theta1 = atan2(v1,x);
        REAL theta2 = atan2(v2,x);

        REAL absZ = fabs(z), signZ;
        if (absZ<1e-10) signZ = 0;
        else            signZ = z/absZ;

        // Loop over gauss points
        REAL thetak, Rtheta, R, expKr, expKz = exp(-kappa*absZ);
        for (int i=0; i<K; i++)
        {
            thetak = (theta2 - theta1)/2*xk[i] + (theta2 + theta1)/2;
            Rtheta = x/cos(thetak);
            R      = sqrt(Rtheta*Rtheta + z*z);
            expKr  = exp(-kappa*R);
            if (kappa>1e-10)
            {
                PHI[0]+= -wk[i]*(expKr - expKz)/kappa * (theta2 - theta1)/2;
                PHI[1]+=  wk[i]*(z/R*expKr - expKz*signZ) * (theta2 - theta1)/2;
            }
            else
            {
                PHI[0]+= wk[i]*(R-absZ) * (theta2 - theta1)/2;
                PHI[1]+= wk[i]*(z/R - signZ) * (theta2 - theta1)/2;
            }

            PHI[2]+= wk[i]*(R-absZ) * (theta2 - theta1)/2;
            PHI[3]+= wk[i]*(z/R - signZ) * (theta2 - theta1)/2;
        }
    }

    __device__ void intSide(REAL *PHI, REAL *v1, REAL *v2, REAL p, REAL kappa, REAL *xk, REAL *wk, int K)
    {
        REAL v21u[3];
        for (int i=0; i<3; i++)
        {
            v21u[i] = v2[i] - v1[i];
        }

        REAL L21 = norm(v21u);
        axip(v21u, 1/L21, 3);

        REAL unit[3] = {0.,0.,1.};
        REAL orthog[3];
        cross(unit, v21u, orthog);

        REAL v1new_x = dot_prod(orthog, v1); 
        REAL v1new_y = dot_prod(v21u, v1); 

        if (v1new_x<0)
        {
            axip(v21u, -1, 3);
            axip(orthog, -1, 3);
            v1new_x = dot_prod(orthog, v1);
            v1new_y = dot_prod(v21u, v1);
        }

        REAL v2new_y = dot_prod(v21u, v2); 

        if ((v1new_y>0 && v2new_y<0) || (v1new_y<0 && v2new_y>0))
        {
            lineInt(PHI, p, v1new_x, 0, v1new_y, kappa, xk, wk, K);
            lineInt(PHI, p, v1new_x, v2new_y, 0, kappa, xk, wk, K);
        }
        else
        {
            REAL PHI_aux[4] = {0.,0.,0.,0.};
            lineInt(PHI_aux, p, v1new_x, v1new_y, v2new_y, kappa, xk, wk, K);

            for(int i=0; i<4; i++)
                PHI[i] -= PHI_aux[i];
        }

    }

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
        // Multiply y_panel times a rotation matrix [X; Y; Z]
        REAL x_aux, y_aux, z_aux;
        x_aux = dot_prod(X, y0_panel);
        y_aux = dot_prod(Y, y0_panel);
        z_aux = dot_prod(Z, y0_panel);
        y0_panel[0] = x_aux;
        y0_panel[1] = y_aux;
        y0_panel[2] = z_aux;

        x_aux = dot_prod(X, y1_panel);
        y_aux = dot_prod(Y, y1_panel);
        z_aux = dot_prod(Z, y1_panel);
        y1_panel[0] = x_aux;
        y1_panel[1] = y_aux;
        y1_panel[2] = z_aux;

        x_aux = dot_prod(X, y2_panel);
        y_aux = dot_prod(Y, y2_panel);
        z_aux = dot_prod(Z, y2_panel);
        y2_panel[0] = x_aux;
        y2_panel[1] = y_aux;
        y2_panel[2] = z_aux;

        x_aux = dot_prod(X, x_panel);
        y_aux = dot_prod(Y, x_panel);
        z_aux = dot_prod(Z, x_panel);
        x_panel[0] = x_aux;
        x_panel[1] = y_aux;
        x_panel[2] = z_aux;

        // Shift origin so it matches collocation point
        for (int i=0; i<2; i++)
        {   
            y0_panel[i] -= x_panel[i]; 
            y1_panel[i] -= x_panel[i]; 
            y2_panel[i] -= x_panel[i]; 
        }   

        // Loop over sides
        intSide(PHI, y0_panel, y1_panel, x_panel[2], kappa, xk, wk, K); // Side 0
        intSide(PHI, y1_panel, y2_panel, x_panel[2], kappa, xk, wk, K); // Side 1
        intSide(PHI, y2_panel, y0_panel, x_panel[2], kappa, xk, wk, K); // Side 2

        if (same==1)
        {
            PHI[1] = -2*M_PI;
            PHI[3] =  2*M_PI;
        }
        
    }

    __global__ void M2P(int *sizeTarDev, int *offsetMltDev,
                        REAL *xtDev, REAL *ytDev, REAL *ztDev,
                        REAL *xcDev, REAL *ycDev, REAL *zcDev, 
                        REAL *mpDev, REAL *mpxDev, REAL *mpyDev, REAL *mpzDev,
                        REAL *Pre0, REAL *Pre1, REAL *Pre2, REAL *Pre3,
                        REAL *p1, REAL *p2, int *Index, int N, REAL kappa, REAL E_hat, int BpT, int NCRIT)
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
        __shared__ int Index_sh[(P+1)*(P+1)*(P+1)];

        for (int ind=0; ind<((P+1)*(P+1)*(P+1)-1)/BSZ; ind++)
        {
            Index_sh[ind*BSZ + threadIdx.x] = Index[ind*BSZ + threadIdx.x];    
        }

        int ind = ((P+1)*(P+1)*(P+1)-1)/BSZ;
        if (threadIdx.x<(P+1)*(P+1)*(P+1)-BSZ*ind)
        {
            Index_sh[ind*BSZ + threadIdx.x] = Index[ind*BSZ + threadIdx.x];
        }
        int i;

        for (int mult=0; mult<Nm; mult++)
        {
            axL[mult] = 0.;
            ayL[mult] = 0.;
            azL[mult] = 0.;
            axY[mult] = 0.;
            ayY[mult] = 0.;
            azY[mult] = 0.;
        }

        for (int iblock=0; iblock<BpT; iblock++)
        {
            i  = I + iblock*BSZ;
            xi = xtDev[i];
            yi = ytDev[i];
            zi = ztDev[i];
            
            REAL L = 0., dL = 0., Y = 0., dY = 0.;

            for(int jblock=0; jblock<(Nmlt-1)/BSZ; jblock++)
            {
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
                        getCoeff(aL, axL, ayL, azL, aY, axY, ayY, azY, 
                                dx, dy, dz, kappa, Index_sh);
                        multipole(L, dL, Y, dY, mpDev, mpxDev, mpyDev, mpzDev, 
                                aL, axL, ayL, azL, aY, axY, ayY, azY, 
                                CJ_start, jblock, j, E_hat);
                    }
                }
            } 

            __syncthreads();
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
                            dx, dy, dz, kappa, Index_sh);
                    multipole(L, dL, Y, dY, mpDev, mpxDev, mpyDev, mpzDev, 
                            aL, axL, ayL, azL, aY, axY, ayY, azY, 
                            CJ_start, jblock, j, E_hat);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
            {
                // With preconditioner
                p1[i] += Pre0[i]*(-dL+L) + Pre1[i]*(dY-Y);
                p2[i] += Pre2[i]*(-dL+L) + Pre3[i]*(dY-Y);

                // No preconditioner
                //p1[i] += -dL + L;
                //p2[i] +=  dY - Y;
            }
        }
        
    }
    
    __global__ void P2P(int *offsetSrcDev, int *offsetIntDev, int *intPtrDev, int *sizeTarDev, int *kDev, 
                        REAL *xsDev, REAL *ysDev, REAL *zsDev, REAL *mDev, REAL *mxDev, 
                        REAL *myDev, REAL *mzDev, REAL *mcleanDev, REAL *xtDev, REAL *ytDev, REAL *ztDev, 
                        REAL *AreaDev, REAL *p1, REAL *p2,REAL *Pre0, REAL *Pre1, REAL *Pre2, 
                        REAL *Pre3, REAL E_hat, int N, REAL *vertexDev, 
                        REAL w0, REAL *xkDev, REAL *wkDev, REAL kappa, 
                        REAL threshold, REAL eps, int BpT, int NCRIT, int *AI_int_gpu)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int CJt_start = offsetIntDev[blockIdx.x];
        int Ntsrc     = offsetIntDev[blockIdx.x+1] - CJt_start;
        
        REAL xi, yi, zi, dx, dy, dz, r, R_tri, dx_tri, dy_tri, dz_tri, L_d;

        __shared__ REAL ver_sh[9*BSZ], xc_sh[BSZ], yc_sh[BSZ], zc_sh[BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mc_sh[BSZ],
                        xk_sh[xkSize], wk_sh[xkSize];


        if (threadIdx.x<xkSize)
        {
            xk_sh[threadIdx.x] = xkDev[threadIdx.x];
            wk_sh[threadIdx.x] = wkDev[threadIdx.x];
        }
        __syncthreads();

        int i, same, CJ_start, Nsrc, twig_ptr;

        for (int iblock=0; iblock<BpT; iblock++)
        {
            REAL L = 0., Y = 0., dL = 0., dY = 0.;
            i  = I + iblock*BSZ;
            xi = xtDev[i];
            yi = ytDev[i];
            zi = ztDev[i];
            int an_counter = 0;

            for (int twig=0; twig<Ntsrc; twig++)
            {
                twig_ptr = intPtrDev[CJt_start + twig];
                CJ_start = offsetSrcDev[twig_ptr];
                Nsrc = offsetSrcDev[twig_ptr+1] - CJ_start;

                for(int jblock=0; jblock<(Nsrc-1)/BSZ; jblock++)
                {
                    __syncthreads();
                    xj_sh[threadIdx.x] = xsDev[CJ_start + jblock*BSZ + threadIdx.x];
                    yj_sh[threadIdx.x] = ysDev[CJ_start + jblock*BSZ + threadIdx.x];
                    zj_sh[threadIdx.x] = zsDev[CJ_start + jblock*BSZ + threadIdx.x];
                    m_sh[threadIdx.x] = mDev[CJ_start + jblock*BSZ + threadIdx.x];
                    mx_sh[threadIdx.x] = mxDev[CJ_start + jblock*BSZ + threadIdx.x];
                    my_sh[threadIdx.x] = myDev[CJ_start + jblock*BSZ + threadIdx.x];
                    mz_sh[threadIdx.x] = mzDev[CJ_start + jblock*BSZ + threadIdx.x];
                    mc_sh[threadIdx.x] = mcleanDev[CJ_start + jblock*BSZ + threadIdx.x];
                    A_sh[threadIdx.x] = AreaDev[CJ_start + jblock*BSZ + threadIdx.x];
                    k_sh[threadIdx.x] = kDev[CJ_start + jblock*BSZ + threadIdx.x];

                    for (int vert=0; vert<9; vert++)
                    {
                        ver_sh[9*threadIdx.x+vert] = vertexDev[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                    }
                    __syncthreads();

                    xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
                    yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
                    zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
                    __syncthreads();

                    if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
                    {
                        for (int j=0; j<BSZ; j++)
                        {
                            dx_tri = xi - xc_sh[j];
                            dy_tri = yi - yc_sh[j];
                            dz_tri = zi - zc_sh[j];
                            R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri + eps);

                            L_d = sqrt(2*A_sh[j])/(R_tri);

                            if(L_d<threshold)
                            {
                                dx = xi - xj_sh[j];
                                dy = yi - yj_sh[j];
                                dz = zi - zj_sh[j];
                                r = sqrt(dx*dx + dy*dy + dz*dz + eps);
                                L  += m_sh[j]/r; 
                                Y  += m_sh[j] * exp(-kappa*r)/r; 
                                dY += -(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*exp(-kappa*r)/(r*r)*(kappa+1/r); 
                                dL += -(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                            }
                            else if(k_sh[j]==0)
                            {
                                REAL PHI_1[4] = {0., 0., 0., 0.};
                                REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                                ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                                ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                                // Check if it is same triangle
                                // same if collocation point matches vertex average

                                dx = xi - xc_sh[j]; 
                                dy = yi - yc_sh[j]; 
                                dz = zi - zc_sh[j]; 
                                r  = sqrt(dx*dx + dy*dy + dz*dz);
                                if (r<1e-8) same=1;
                                else        same=0;
                                

                                SA(PHI_1, panel, xi, yi, zi, kappa, same, xk_sh, wk_sh, xkSize);
                    
                                Y  += PHI_1[0] * m_sh[j]/(w0*A_sh[j]);
                                L  += PHI_1[2] * m_sh[j]/(w0*A_sh[j]);
                                dL += PHI_1[3] * mc_sh[j];
                                dY += PHI_1[1] * mc_sh[j];
                                an_counter += 1;
                            }
                        }
                    }
                }
               
                __syncthreads();
                int jblock = (Nsrc-1)/BSZ;
                xj_sh[threadIdx.x] = xsDev[CJ_start + jblock*BSZ + threadIdx.x];
                yj_sh[threadIdx.x] = ysDev[CJ_start + jblock*BSZ + threadIdx.x];
                zj_sh[threadIdx.x] = zsDev[CJ_start + jblock*BSZ + threadIdx.x];
                m_sh[threadIdx.x] = mDev[CJ_start + jblock*BSZ + threadIdx.x];
                mx_sh[threadIdx.x] = mxDev[CJ_start + jblock*BSZ + threadIdx.x];
                my_sh[threadIdx.x] = myDev[CJ_start + jblock*BSZ + threadIdx.x];
                mz_sh[threadIdx.x] = mzDev[CJ_start + jblock*BSZ + threadIdx.x];
                mc_sh[threadIdx.x] = mcleanDev[CJ_start + jblock*BSZ + threadIdx.x];
                A_sh[threadIdx.x] = AreaDev[CJ_start + jblock*BSZ + threadIdx.x];
                k_sh[threadIdx.x] = kDev[CJ_start + jblock*BSZ + threadIdx.x];

                for (int vert=0; vert<9; vert++)
                {
                    ver_sh[9*threadIdx.x+vert] = vertexDev[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                }
                __syncthreads();

                xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
                yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
                zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
                __syncthreads();
                if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
                {
                    for (int j=0; j<Nsrc-(jblock*BSZ); j++)
                    {
                        dx_tri = xi - xc_sh[j];
                        dy_tri = yi - yc_sh[j];
                        dz_tri = zi - zc_sh[j];
                        R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri + eps);

                        L_d = sqrt(2*A_sh[j])/(R_tri);

                        if (L_d<threshold)
                        {
                            dx = xi - xj_sh[j];
                            dy = yi - yj_sh[j];
                            dz = zi - zj_sh[j];
                            r = sqrt(dx*dx + dy*dy + dz*dz + eps);
                            L  += m_sh[j]/r; 
                            Y  += m_sh[j] * exp(-kappa*r)/r; 
                            dY += -(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*exp(-kappa*r)/(r*r)*(kappa+1/r); 
                            dL += -(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                        }

                        else if(k_sh[j]==0)
                        {
                            REAL PHI_1[4] = {0., 0., 0., 0.};
                            REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                            ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                            ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                            // Check if it is same triangle
                            // same if collocation point matches vertex average

                            dx = xi - xc_sh[j]; 
                            dy = yi - yc_sh[j]; 
                            dz = zi - zc_sh[j]; 
                            r  = sqrt(dx*dx + dy*dy + dz*dz);
                            if (r<1e-8) same=1;
                            else        same=0;

                            SA(PHI_1, panel, xi, yi, zi, kappa, same, xk_sh, wk_sh, xkSize);
                
                            Y  += PHI_1[0] * m_sh[j]/(w0*A_sh[j]);
                            L  += PHI_1[2] * m_sh[j]/(w0*A_sh[j]);
                            dL += PHI_1[3] * mc_sh[j];
                            dY += PHI_1[1] * mc_sh[j];
                            an_counter += 1;
                        }
                    }
                }
            }
        
            if (threadIdx.x+iblock*BSZ<sizeTarDev[blockIdx.x])
            {
                // With preconditioner
                p1[i] += Pre0[i]*(-dL+L) + Pre1[i]*(dY-E_hat*Y);
                p2[i] += Pre2[i]*(-dL+L) + Pre3[i]*(dY-E_hat*Y);

                // No preconditioner
                //p1[i] += -dL + L;
                //p2[i] +=  dY - E_hat*Y;
                AI_int_gpu[i] = an_counter;
            }
        }
    }

    __global__ void get_phir(REAL *phir, REAL *xs, REAL *ys, REAL *zs, 
                            REAL *m, REAL *mx, REAL *my, REAL *mz, REAL *mclean, 
                            REAL *xq, REAL *yq, REAL *zq, 
                            REAL *Area, int *k, REAL *vertex, 
                            int N, int Nj, int Nq, int K,
                            REAL w0, REAL *xk, REAL *wk, 
                            REAL threshold, int *AI_int_gpu)
    {
        int i = threadIdx.x + blockIdx.x*BSZ;
        REAL xi, yi, zi, dx, dy, dz, r, L_d;
        int jblock, triangle;

        __shared__ REAL ver_sh[9*BSZ], xc_sh[BSZ], yc_sh[BSZ], zc_sh[BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mc_sh[BSZ],
                        xk_sh[xkSize], wk_sh[xkSize];


        if (threadIdx.x<xkSize)
        {
            xk_sh[threadIdx.x] = xk[threadIdx.x];
            wk_sh[threadIdx.x] = wk[threadIdx.x];
        }
        __syncthreads();

        REAL L = 0., Y = 0., dL = 0., dY = 0.;
        xi = xq[i];
        yi = yq[i];
        zi = zq[i];
        int an_counter = 0;

        for(jblock=0; jblock<(Nj-1)/BSZ; jblock++)
        {   
            __syncthreads();
            xj_sh[threadIdx.x] = xs[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = ys[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zs[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mc_sh[threadIdx.x] = mclean[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[(jblock*BSZ + threadIdx.x)/K];
            
            for (int vert=0; vert<9; vert++)
            {
                triangle = (jblock*BSZ+threadIdx.x)/K;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
            __syncthreads();
            
            xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
            yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
            zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
            __syncthreads();

            for (int j=0; j<BSZ; j++)
            {
                dx = xi - xc_sh[j];
                dy = yi - yc_sh[j];
                dz = zi - zc_sh[j];
                r  = sqrt(dx*dx + dy*dy + dz*dz);

                L_d = sqrt(2*A_sh[j])/r;

                if(L_d<threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    r = sqrt(dx*dx + dy*dy + dz*dz);
                    L  += m_sh[j]/r; 
                    Y  += 0;
                    dY += 0;
                    dL += -(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                }
                else if(k_sh[j]==0)
                {
                    REAL PHI_1[4] = {0., 0., 0., 0.};
                    REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                    ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                    ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                    SA(PHI_1, panel, xi, yi, zi, 1e-15, 0, xk_sh, wk_sh, xkSize);
        
                    Y  += 0;
                    L  += PHI_1[2] * m_sh[j]/(w0*A_sh[j]);
                    dL += PHI_1[3] * mc_sh[j];
                    dY += 0;
                    an_counter += 1;
                }
            }
        }
    
        __syncthreads();
        jblock = (Nj-1)/BSZ;
        if (threadIdx.x<Nj-jblock*BSZ)
        {
            xj_sh[threadIdx.x] = xs[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = ys[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zs[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mc_sh[threadIdx.x] = mclean[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[(jblock*BSZ + threadIdx.x)/K];

            for (int vert=0; vert<9; vert++)
            {
                triangle = (jblock*BSZ+threadIdx.x)/K;
                ver_sh[9*threadIdx.x+vert] = vertex[9*triangle+vert];
            }
        }
        __syncthreads();

        if (threadIdx.x<Nj-jblock*BSZ)
        {
            xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
            yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
            zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
        }
        __syncthreads();

        for (int j=0; j<Nj-(jblock*BSZ); j++)
        {
            dx = xi - xc_sh[j];
            dy = yi - yc_sh[j];
            dz = zi - zc_sh[j];
            r  = sqrt(dx*dx + dy*dy + dz*dz);

            L_d = sqrt(2*A_sh[j])/r;

            if (L_d<threshold)
            {
                dx = xi - xj_sh[j];
                dy = yi - yj_sh[j];
                dz = zi - zj_sh[j];
                r = sqrt(dx*dx + dy*dy + dz*dz);
                L  += m_sh[j]/r; 
                Y  += 0;
                dY += 0;
                dL += -(mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
            }

            else if(k_sh[j]==0)
            {
                REAL PHI_1[4] = {0., 0., 0., 0.};
                REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                SA(PHI_1, panel, xi, yi, zi, 1e-15, 0, xk_sh, wk_sh, xkSize);
    
                Y  += 0;
                L  += PHI_1[2] * m_sh[j]/(w0*A_sh[j]);
                dL += PHI_1[3] * mc_sh[j];
                dY += 0;
                an_counter += 1;
            }
        }
       
        if (i<Nq)
        {
            phir[i] = L - dL;
            AI_int_gpu[i] = an_counter;
        }
    }

    __global__ void compute_RHS(REAL *F, REAL *xq, REAL *yq, REAL *zq,
                                REAL *q, REAL *xi, REAL *yi, REAL *zi,
                                REAL *P0, REAL *P2, int N, int Nq, REAL E_1)
    {
        int I = blockIdx.x*BSZ + threadIdx.x;
        REAL x = xi[I], y = yi[I], z = zi[I], sum=0.;
        REAL dx, dy, dz, r;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ], q_sh[BSZ];

        for (int block=0; block<(Nq-1)/BSZ; block++)
        {
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
            __syncthreads();

            for (int i=0; i<BSZ; i++)
            {
                dx = xq_sh[i] - x;
                dy = yq_sh[i] - y;
                dz = zq_sh[i] - z;
                r  = sqrt(dx*dx + dy*dy + dz*dz);

                sum += -q_sh[i]/(E_1*r);
            }
        }

        int block = (Nq-1)/BSZ; 
        __syncthreads();
        xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
        yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
        zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
        q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
        __syncthreads();

        for (int i=0; i<Nq-block*BSZ; i++)
        {
            dx = xq_sh[i] - x;
            dy = yq_sh[i] - y;
            dz = zq_sh[i] - z;
            r  = sqrt(dx*dx + dy*dy + dz*dz);

            sum += -q_sh[i]/(E_1*r);
        }

        if (I<N)
        {
            F[I] = sum*P0[I];
            F[I+N] = sum*P2[I];
        }

    }

    
    """%{'blocksize':BSZ, 'Nmult':Nm, 'K_1D':xkSize, 'Ptree':P}, nvcc="nvcc", options=["-use_fast_math","-Xptxas=-v,-abi=no"])

    return mod
