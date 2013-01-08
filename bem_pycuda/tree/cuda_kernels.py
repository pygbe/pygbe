from pycuda.compiler import SourceModule

def kernels(BSZ, Nm, xkSize, P, REAL):
    
    mod = SourceModule( """

    #define REAL %(precision)s
    #define BSZ %(blocksize)d
    #define Nm  %(Nmult)d
    #define xkSize %(K_1D)d
    #define P      %(Ptree)d

    /*
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
    */

    __device__ int getIndex(int i, int j, int k, int *Index)
    {   
        return Index[(P+1)*(P+1)*i + (P+1)*j + k]; 
    }


    __device__ void getCoeff(REAL *a, REAL dx, REAL dy, REAL dz, REAL kappa, int *index, int LorY)
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
        I = getIndex(1,0,0, index);

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
            I    = getIndex(i,0,0, index);
            Im1x = getIndex(i-1,0,0, index);
            Im2x = getIndex(i-2,0,0, index);
            if (LorY==2) // if Yukawa
            {   
                b[I] = Cb * (dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dx*b[Im1x] + b[Im2x]) -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
            }

            if (LorY==1) // if Laplace
            {   
                a[I] = C * ( -(2*i-1)*dx*a[Im1x] - (i-1)*a[Im2x] );
            }

            I    = getIndex(0,i,0, index);
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
        I    = getIndex(1,1,0, index);
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

        I    = getIndex(1,0,1, index);
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

        I    = getIndex(0,1,1, index);
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
            I    = getIndex(1,i,0, index);
            Im1x = getIndex(0,i,0, index);
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

            I    = getIndex(1,0,i, index);
            Im1x = getIndex(0,0,i, index);
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

            I    = getIndex(0,1,i, index);
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

            I    = getIndex(i,1,0, index);
            Im1y = I-(P+2-1-i);
            Im1x = getIndex(i-1,1,0, index);
            Im2x = getIndex(i-2,1,0, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dy*a[Im1y] + dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dy*b[Im1y]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dy*a[Im1y]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            I    = getIndex(i,0,1, index);
            Im1z = I-1;
            Im1x = getIndex(i-1,0,1, index);
            Im2x = getIndex(i-2,0,1, index);
            if (LorY==2) // if Yukawa
            {
                b[I] = Cb * (dz*a[Im1z] + dx*a[Im1x] + a[Im2x]);
                a[I] = C * ( -kappa*(dz*b[Im1z]+dx*b[Im1x]+b[Im2x]) -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }
            if (LorY==1) // if Laplace
            {
                a[I] = C * ( -(2*(1+i)-1)*(dz*a[Im1z]+dx*a[Im1x]) - (1+i-1)*(a[Im2x]) );
            }

            I    = getIndex(0,i,1, index);
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
                I    = getIndex(i,j,0, index);
                Im1x = getIndex(i-1,j,0, index);
                Im2x = getIndex(i-2,j,0, index);
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

                I    = getIndex(i,0,j, index);
                Im1x = getIndex(i-1,0,j, index);
                Im2x = getIndex(i-2,0,j, index);
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

                I    = getIndex(0,i,j, index);
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
            I    = getIndex(1,1,1, index);
            Im1x = getIndex(0,1,1, index);
            Im1y = getIndex(1,0,1, index);
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
                I    = getIndex(i,1,1, index);
                Im1x = getIndex(i-1,1,1, index);
                Im1y = I-(P+2-i-1);
                Im1z = I-1;
                Im2x = getIndex(i-2,1,1, index);
                if (LorY==2) // if Yukawa
                {
                    b[I] = Cb * (dx*a[Im1x] + dy*a[Im1y] + dz*a[Im1z] + a[Im2x]);
                    a[I] = C * ( -kappa*(dx*b[Im1x]+dy*b[Im1y]+dz*b[Im1z]+b[Im2x]) -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
                }
                if (LorY==1) // if Laplace
                {
                    a[I] = C * ( -(2*(i+2)-1)*(dx*a[Im1x]+dy*a[Im1y]+dz*a[Im1z]) - (i+1)*(a[Im2x]) );
                }

                I    = getIndex(1,i,1, index);
                Im1x = getIndex(0,i,1, index);
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


                I    = getIndex(1,1,i, index);
                Im1x = getIndex(0,1,i, index);
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
                    I    = getIndex(1,i,j, index);
                    Im1x = getIndex(0,i,j, index);
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

                    I    = getIndex(i,1,j, index);
                    Im1x = getIndex(i-1,1,j, index);
                    Im1y = I-(P+2-i-1);
                    Im2x = getIndex(i-2,1,j, index);
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

                    I    = getIndex(i,j,1, index);
                    Im1x = getIndex(i-1,j,1, index);
                    Im2x = getIndex(i-2,j,1, index);
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
                        I    = getIndex(i,j,k, index);
                        Im1x = getIndex(i-1,j,k, index);
                        Im2x = getIndex(i-2,j,k, index);
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

    __device__ void multipole(REAL &K, REAL &V, REAL *M, REAL *Md,
                            REAL *a, int CJ_start, int jblock, int j)
    {
        int offset;
        for (int i=0; i<Nm; i++)
        {
            offset = (CJ_start+j)*Nm + jblock*BSZ*Nm + i;
            V += M[offset] * a[i];
            K += Md[offset]* a[i]; 
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

    __device__ void lineInt(REAL &PHI_K, REAL &PHI_V, REAL z, REAL x, REAL v1, REAL v2, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
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
            if (LorY==2)
            {
                if (kappa>1e-12)
                {
                    PHI_V+= -wk[i]*(expKr - expKz)/kappa * (theta2 - theta1)/2;
                    PHI_K+=  wk[i]*(z/R*expKr - expKz*signZ) * (theta2 - theta1)/2;
                }
                else
                {
                    PHI_V+= wk[i]*(R-absZ) * (theta2 - theta1)/2;
                    PHI_K+= wk[i]*(z/R - signZ) * (theta2 - theta1)/2;
                }
            }

            if (LorY==1)
            {
                PHI_V += wk[i]*(R-absZ) * (theta2 - theta1)/2;
                PHI_K += wk[i]*(z/R - signZ) * (theta2 - theta1)/2;
            }
        }
    }

    __device__ void intSide(REAL &PHI_K, REAL &PHI_V, REAL *v1, REAL *v2, REAL p, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
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
            lineInt(PHI_K, PHI_V, p, v1new_x, 0, v1new_y, kappa, xk, wk, K, LorY);
            lineInt(PHI_K, PHI_V, p, v1new_x, v2new_y, 0, kappa, xk, wk, K, LorY);

        }
        else
        {
            REAL PHI_Kaux = 0., PHI_Vaux = 0.;
            lineInt(PHI_Kaux, PHI_Vaux, p, v1new_x, v1new_y, v2new_y, kappa, xk, wk, K, LorY);

            PHI_K -= PHI_Kaux;
            PHI_V -= PHI_Vaux;
        }
    }

    __device__ void SA(REAL &PHI_K, REAL &PHI_V, REAL *y, REAL x0, REAL x1, REAL x2, 
                       REAL K_diag, REAL V_diag, REAL kappa, int same, REAL *xk, REAL *wk, int K, int LorY)
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
        intSide(PHI_K, PHI_V, y0_panel, y1_panel, x_panel[2], kappa, xk, wk, K, LorY); // Side 0
        intSide(PHI_K, PHI_V, y1_panel, y2_panel, x_panel[2], kappa, xk, wk, K, LorY); // Side 1
        intSide(PHI_K, PHI_V, y2_panel, y0_panel, x_panel[2], kappa, xk, wk, K, LorY); // Side 2

        if (same==1)
        {
            PHI_K += K_diag;
            PHI_V += V_diag;
        }
        
    }

    __global__ void M2P(REAL *K_gpu, REAL *V_gpu, int *offMlt, int *sizeTar, REAL *xc, REAL *yc, REAL *zc, 
                        REAL *M, REAL *Md, REAL *xt, REAL *yt, REAL *zt,
                        int *Index, int ptr_off, int ptr_lst, REAL kappa, int BpT, int NCRIT, int LorY)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int CJ_start = offMlt[ptr_off+blockIdx.x];
        int Nmlt     = offMlt[ptr_off+blockIdx.x+1] - CJ_start;


        REAL xi, yi, zi,
             dx, dy, dz; 
        REAL a[Nm];

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


        for (int iblock=0; iblock<BpT; iblock++)
        {
            i  = I + iblock*BSZ;
            xi = xt[i];
            yi = yt[i];
            zi = zt[i];
            
            REAL K = 0., V = 0.;

            for(int jblock=0; jblock<(Nmlt-1)/BSZ; jblock++)
            {
                __syncthreads();
                xc_sh[threadIdx.x] = xc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                yc_sh[threadIdx.x] = yc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                zc_sh[threadIdx.x] = zc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int j=0; j<BSZ; j++)
                    {
                        dx = xi - xc_sh[j];
                        dy = yi - yc_sh[j];
                        dz = zi - zc_sh[j];
                        getCoeff(a, dx, dy, dz, 
                                kappa, Index_sh, LorY);
                        multipole(K, V, M, Md, a,
                                CJ_start, jblock, j);
                    }
                }
            } 

            __syncthreads();
            int jblock = (Nmlt-1)/BSZ;
            xc_sh[threadIdx.x] = xc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            yc_sh[threadIdx.x] = yc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            zc_sh[threadIdx.x] = zc[ptr_lst + CJ_start + jblock*BSZ + threadIdx.x];
            __syncthreads();
            
            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int j=0; j<Nmlt-(jblock*BSZ); j++)
                {
                    dx = xi - xc_sh[j];
                    dy = yi - yc_sh[j];
                    dz = zi - zc_sh[j];
                    getCoeff(a, dx, dy, dz, 
                            kappa, Index_sh, LorY);
                    multipole(K, V, M, Md, a,
                            CJ_start, jblock, j);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                K_gpu[i] += K;
                V_gpu[i] += V; 
            }
        }
        
    }
    
    __global__ void P2P(REAL *K_gpu, REAL *V_gpu, int *offSrc, int *offTwg, int *P2P_list, int *sizeTar, int *k, 
                        REAL *xj, REAL *yj, REAL *zj, REAL *m, REAL *mx, REAL *my, REAL *mz, REAL *mKc, REAL *mVc, 
                        REAL *xt, REAL *yt, REAL *zt, REAL *Area, REAL *vertex, REAL *xk, REAL *wk, int ptr_off, 
                        int ptr_lst, int LorY, REAL kappa, REAL threshold, REAL eps, int BpT, int NCRIT, 
                        REAL K_diag, REAL V_diag, int *AI_int_gpu)
    {
        int I = threadIdx.x + blockIdx.x*NCRIT;
        int list_start = offTwg[ptr_off+blockIdx.x];
        int list_end   = offTwg[ptr_off+blockIdx.x+1];
        
        REAL xi, yi, zi, dx, dy, dz, r, R_tri, dx_tri, dy_tri, dz_tri, L_d;

        __shared__ REAL ver_sh[9*BSZ], xc_sh[BSZ], yc_sh[BSZ], zc_sh[BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mKc_sh[BSZ], mVc_sh[BSZ],
                        xk_sh[xkSize], wk_sh[xkSize];


        if (threadIdx.x<xkSize)
        {
            xk_sh[threadIdx.x] = xk[threadIdx.x];
            wk_sh[threadIdx.x] = wk[threadIdx.x];
        }
        __syncthreads();

        int i, same, CJ_start, Nsrc, CJ;

        for (int iblock=0; iblock<BpT; iblock++)
        {
            REAL sum_K = 0., sum_V = 0.;
            i  = I + iblock*BSZ;
            xi = xt[i];
            yi = yt[i];
            zi = zt[i];
            int an_counter = 0;

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = P2P_list[ptr_lst+lst];
                CJ_start = offSrc[CJ];
                Nsrc = offSrc[CJ+1] - CJ_start;

                for(int jblock=0; jblock<(Nsrc-1)/BSZ; jblock++)
                {
                    __syncthreads();
                    xj_sh[threadIdx.x] = xj[CJ_start + jblock*BSZ + threadIdx.x];
                    yj_sh[threadIdx.x] = yj[CJ_start + jblock*BSZ + threadIdx.x];
                    zj_sh[threadIdx.x] = zj[CJ_start + jblock*BSZ + threadIdx.x];
                    m_sh[threadIdx.x]  = m[CJ_start + jblock*BSZ + threadIdx.x];
                    mx_sh[threadIdx.x] = mx[CJ_start + jblock*BSZ + threadIdx.x];
                    my_sh[threadIdx.x] = my[CJ_start + jblock*BSZ + threadIdx.x];
                    mz_sh[threadIdx.x] = mz[CJ_start + jblock*BSZ + threadIdx.x];
                    mKc_sh[threadIdx.x] = mKc[CJ_start + jblock*BSZ + threadIdx.x];
                    mVc_sh[threadIdx.x] = mVc[CJ_start + jblock*BSZ + threadIdx.x];
                    A_sh[threadIdx.x]  = Area[CJ_start + jblock*BSZ + threadIdx.x];
                    k_sh[threadIdx.x]  = k[CJ_start + jblock*BSZ + threadIdx.x];

                    for (int vert=0; vert<9; vert++)
                    {
                        ver_sh[9*threadIdx.x+vert] = vertex[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                    }
                    __syncthreads();

                    xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
                    yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
                    zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
                    __syncthreads();

                    if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
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
                                if (LorY==2)
                                {
                                    sum_V += m_sh[j] * exp(-kappa*r)/r; 
                                    sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*exp(-kappa*r)/(r*r)*(kappa+1/r); 

                                }
                                if (LorY==1)
                                {
                                    sum_V += m_sh[j]/r; 
                                    sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                                }
                            }

                            else if(k_sh[j]==0)
                            {
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
                                
                                REAL PHI_K=0., PHI_V=0.;

                                SA(PHI_K, PHI_V, panel, xi, yi, zi, K_diag, V_diag, 
                                    kappa, same, xk_sh, wk_sh, xkSize, LorY);
                    
                                sum_V += PHI_V * mVc_sh[j];
                                sum_K += PHI_K * mKc_sh[j];
                                an_counter += 1;
                            }
                        }
                    }
                }
                __syncthreads();
                int jblock = (Nsrc-1)/BSZ;
                xj_sh[threadIdx.x] = xj[CJ_start + jblock*BSZ + threadIdx.x];
                yj_sh[threadIdx.x] = yj[CJ_start + jblock*BSZ + threadIdx.x];
                zj_sh[threadIdx.x] = zj[CJ_start + jblock*BSZ + threadIdx.x];
                m_sh[threadIdx.x] = m[CJ_start + jblock*BSZ + threadIdx.x];
                mx_sh[threadIdx.x] = mx[CJ_start + jblock*BSZ + threadIdx.x];
                my_sh[threadIdx.x] = my[CJ_start + jblock*BSZ + threadIdx.x];
                mz_sh[threadIdx.x] = mz[CJ_start + jblock*BSZ + threadIdx.x];
                mKc_sh[threadIdx.x] = mKc[CJ_start + jblock*BSZ + threadIdx.x];
                mVc_sh[threadIdx.x] = mVc[CJ_start + jblock*BSZ + threadIdx.x];
                A_sh[threadIdx.x] = Area[CJ_start + jblock*BSZ + threadIdx.x];
                k_sh[threadIdx.x] = k[CJ_start + jblock*BSZ + threadIdx.x];

                for (int vert=0; vert<9; vert++)
                {
                    ver_sh[9*threadIdx.x+vert] = vertex[9*(CJ_start+jblock*BSZ+threadIdx.x)+vert];
                }
                __syncthreads();

                xc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x] + ver_sh[9*threadIdx.x+3] + ver_sh[9*threadIdx.x+6])/3;
                yc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+1] + ver_sh[9*threadIdx.x+4] + ver_sh[9*threadIdx.x+7])/3;
                zc_sh[threadIdx.x] = (ver_sh[9*threadIdx.x+2] + ver_sh[9*threadIdx.x+5] + ver_sh[9*threadIdx.x+8])/3;
                __syncthreads();
                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
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

                            if (LorY==2)
                            {
                                sum_V += m_sh[j] * exp(-kappa*r)/r; 
                                sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)*exp(-kappa*r)/(r*r)*(kappa+1/r); 

                            }
                            if (LorY==1)
                            {
                                sum_V += m_sh[j]/r; 
                                sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                            }
                        }

                        else if(k_sh[j]==0)
                        {
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

                            REAL PHI_K=0., PHI_V=0.;

                            SA(PHI_K, PHI_V, panel, xi, yi, zi, K_diag, V_diag, 
                                    kappa, same, xk_sh, wk_sh, xkSize, LorY);
                
                            sum_V += PHI_V * mVc_sh[j];
                            sum_K += PHI_K * mKc_sh[j];
                            an_counter += 1;
                        }
                    }
                }
            }
        
            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                K_gpu[i] += sum_K;
                V_gpu[i] += sum_V; 

                AI_int_gpu[i] = an_counter;
            }
        }
    }
   
    __global__ void get_phir(REAL *phir, REAL *xq, REAL *yq, REAL *zq,
                            REAL *m, REAL *mx, REAL *my, REAL *mz, REAL *mKc, REAL *mVc, 
                            REAL *xj, REAL *yj, REAL *zj, REAL *Area, int *k, REAL *vertex, 
                            int Nj, int Nq, int K, REAL *xk, REAL *wk, 
                            REAL threshold, int *AI_int_gpu, int Nk)
    {
        int i = threadIdx.x + blockIdx.x*BSZ;
        REAL xi, yi, zi, dx, dy, dz, r, L_d;
        int jblock, triangle;

        __shared__ REAL ver_sh[9*BSZ], xc_sh[BSZ], yc_sh[BSZ], zc_sh[BSZ],
                        xj_sh[BSZ], yj_sh[BSZ], zj_sh[BSZ], A_sh[BSZ], k_sh[BSZ],
                        m_sh[BSZ], mx_sh[BSZ], my_sh[BSZ], mz_sh[BSZ], mKc_sh[BSZ],
                        mVc_sh[BSZ];



        REAL sum_V = 0., sum_K = 0.;
        xi = xq[i];
        yi = yq[i];
        zi = zq[i];
        int an_counter = 0;

        for(jblock=0; jblock<(Nj-1)/BSZ; jblock++)
        {   
            __syncthreads();
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[(jblock*BSZ + threadIdx.x)];
            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
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
                    sum_V  += m_sh[j]/r; 
                    sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                }
                else if(k_sh[j]==0)
                {
                    REAL PHI_K = 0.;
                    REAL PHI_V = 0.;
                    REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                    ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                    ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                    SA(PHI_K, PHI_V, panel, xi, yi, zi, 0, 0, 
                        1e-15, 0, xk, wk, Nk, 1);
        
                    sum_V += PHI_V * mVc_sh[j];
                    sum_K += PHI_K * mKc_sh[j];
                    an_counter += 1;
                }
            }
        }
    
        __syncthreads();
        jblock = (Nj-1)/BSZ;
        if (threadIdx.x<Nj-jblock*BSZ)
        {
            xj_sh[threadIdx.x] = xj[jblock*BSZ + threadIdx.x];
            yj_sh[threadIdx.x] = yj[jblock*BSZ + threadIdx.x];
            zj_sh[threadIdx.x] = zj[jblock*BSZ + threadIdx.x];
            m_sh[threadIdx.x]  = m[jblock*BSZ + threadIdx.x];
            mx_sh[threadIdx.x] = mx[jblock*BSZ + threadIdx.x];
            my_sh[threadIdx.x] = my[jblock*BSZ + threadIdx.x];
            mz_sh[threadIdx.x] = mz[jblock*BSZ + threadIdx.x];
            mKc_sh[threadIdx.x] = mKc[jblock*BSZ + threadIdx.x];
            mVc_sh[threadIdx.x] = mVc[jblock*BSZ + threadIdx.x];
            k_sh[threadIdx.x]  = k[jblock*BSZ + threadIdx.x];
            A_sh[threadIdx.x]  = Area[jblock*BSZ + threadIdx.x];

            for (int vert=0; vert<9; vert++)
            {
                triangle = jblock*BSZ+threadIdx.x;
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

            if (i<Nq)
            {
                if (L_d<threshold)
                {
                    dx = xi - xj_sh[j];
                    dy = yi - yj_sh[j];
                    dz = zi - zj_sh[j];
                    r = sqrt(dx*dx + dy*dy + dz*dz);
                    sum_V  += m_sh[j]/r; 
                    sum_K += (mx_sh[j]*dx+my_sh[j]*dy+mz_sh[j]*dz)/(r*r*r);
                }

                else if(k_sh[j]==0)
                {
                    REAL PHI_K = 0.;
                    REAL PHI_V = 0.;
                    REAL panel[9] = {ver_sh[9*j], ver_sh[9*j+1], ver_sh[9*j+2],
                                    ver_sh[9*j+3], ver_sh[9*j+4], ver_sh[9*j+5],
                                    ver_sh[9*j+6], ver_sh[9*j+7], ver_sh[9*j+8]};

                    SA(PHI_K, PHI_V, panel, xi, yi, zi, 0, 0, 
                            1e-15, 0, xk, wk, Nk, 1);
        
                    sum_V += PHI_V * mVc_sh[j];
                    sum_K += PHI_K * mKc_sh[j];
                    
                    an_counter += 1;
                }
            }
        }
       
        if (i<Nq)
        {
            phir[i] = (-sum_K + sum_V)/(4*M_PI);
            AI_int_gpu[i] = an_counter;
        }
    }

    __global__ void compute_RHS(REAL *F, REAL *xq, REAL *yq, REAL *zq,
                                REAL *q, REAL *xi, REAL *yi, REAL *zi,
                                int *sizeTar, int Nq, REAL E_1, 
                                int NCRIT, int BpT)
    {
        int II = threadIdx.x + blockIdx.x*NCRIT;
        int I;
        REAL x, y, z, sum;
        REAL dx, dy, dz, r;
        __shared__ REAL xq_sh[BSZ], yq_sh[BSZ], zq_sh[BSZ], q_sh[BSZ];

        for (int iblock=0; iblock<BpT; iblock++)
        {
            I = II + iblock*BSZ;
            x = xi[I];
            y = yi[I];
            z = zi[I];
            sum = 0.;

            for (int block=0; block<(Nq-1)/BSZ; block++)
            {
                __syncthreads();
                xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
                yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
                zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
                q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
                __syncthreads();

                if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
                {
                    for (int i=0; i<BSZ; i++)
                    {
                        dx = xq_sh[i] - x;
                        dy = yq_sh[i] - y;
                        dz = zq_sh[i] - z;
                        r  = sqrt(dx*dx + dy*dy + dz*dz);

                        sum += q_sh[i]/(E_1*r);
                    }
                }
            }

            int block = (Nq-1)/BSZ; 
            __syncthreads();
            xq_sh[threadIdx.x] = xq[block*BSZ+threadIdx.x];
            yq_sh[threadIdx.x] = yq[block*BSZ+threadIdx.x];
            zq_sh[threadIdx.x] = zq[block*BSZ+threadIdx.x];
            q_sh[threadIdx.x]  = q[block*BSZ+threadIdx.x];
            __syncthreads();

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                for (int i=0; i<Nq-block*BSZ; i++)
                {
                    dx = xq_sh[i] - x;
                    dy = yq_sh[i] - y;
                    dz = zq_sh[i] - z;
                    r  = sqrt(dx*dx + dy*dy + dz*dz);

                    sum += q_sh[i]/(E_1*r);
                }
            }

            if (threadIdx.x+iblock*BSZ<sizeTar[blockIdx.x])
            {
                F[I] = sum;
            }
        }
    }

    
    """%{'blocksize':BSZ, 'Nmult':Nm, 'K_1D':xkSize, 'Ptree':P, 'precision':REAL}, nvcc="nvcc", options=["-use_fast_math","-Xptxas=-v,-abi=no"])

    return mod
