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
#include <sys/time.h>
#define REAL double

double get_time (void)
{
    struct timeval tv; 
    gettimeofday(&tv,NULL);
    return (double)(tv.tv_sec+1e-6*tv.tv_usec);
}

REAL norm(REAL *x)
{
    return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}

void cross(REAL *x, REAL *y, REAL *z) // z is the resulting array
{
    z[0] = x[1]*y[2] - x[2]*y[1];
    z[1] = x[2]*y[0] - x[0]*y[2];
    z[2] = x[0]*y[1] - x[1]*y[0];
}

void MV(REAL *M, REAL *V, REAL *res) // 3x3 mat-vec
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

REAL dot_prod(REAL *x, REAL *y) // len(3) vector dot product
{
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

void axpy(REAL *x, REAL *y, REAL *z, REAL alpha, int sign, int N)
{
    for(int i=0; i<N; i++)
    {
        z[i] = sign*alpha*x[i] + y[i];
    }
}

void ax(REAL *x, REAL *y, REAL alpha, int N)
{
    for(int i=0; i<N; i++)
    {
        y[i] = alpha*x[i];
    }

}

void lineInt(REAL &PHI_K, REAL &PHI_V, REAL z, REAL x, REAL v1, REAL v2, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
{
    REAL theta1 = atan2(v1,x);
    REAL theta2 = atan2(v2,x);
    REAL dtheta = theta2 - theta1;
    REAL thetam = (theta2 + theta1)/2; 


    REAL absZ = fabs(z), signZ;
    if (absZ<1e-10) signZ = 0;
    else            signZ = z/absZ;

    // Loop over gauss points
    REAL thetak, Rtheta, R, expKr, expKz;
    if (LorY==2)
        expKz = exp(-kappa*absZ);

    for (int i=0; i<K; i++)
    {
        thetak = dtheta/2*xk[i] + thetam;
        Rtheta = x/cos(thetak);
        R      = sqrt(Rtheta*Rtheta + z*z);
        expKr  = exp(-kappa*R);
        if (LorY==2)
        {
            if (kappa>1e-12)
            {
                PHI_V += -wk[i]*(expKr - expKz)/kappa * dtheta/2;
                PHI_K +=  wk[i]*(z/R*expKr - expKz*signZ) * dtheta/2;
            }
            else
            {
                PHI_V +=  wk[i]*(R-absZ) * dtheta/2;
                PHI_K +=  wk[i]*(z/R - signZ) * dtheta/2;
            }
        }
        if (LorY==1)
        {
            PHI_V +=  wk[i]*(R-absZ) * dtheta/2;
            PHI_K +=  wk[i]*(z/R - signZ) * dtheta/2;
        }
    }
}

void intSide(REAL &PHI_K, REAL &PHI_V, REAL *v1, REAL *v2, REAL p, REAL kappa, REAL *xk, REAL *wk, int K, int LorY)
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
        REAL PHI1_K = 0. , PHI2_K = 0.;
        REAL PHI1_V = 0. , PHI2_V = 0.;
        lineInt(PHI1_K, PHI1_V, p, x, 0, v1new[1], kappa, xk, wk, K, LorY);
        lineInt(PHI2_K, PHI2_V, p, x, v2new[1], 0, kappa, xk, wk, K, LorY);

        PHI_K += PHI1_K + PHI2_K;
        PHI_V += PHI1_V + PHI2_V;
    }   
    else
    {
        REAL PHI_Kaux = 0., PHI_Vaux = 0.;
        lineInt(PHI_Kaux, PHI_Vaux, p, x, v1new[1], v2new[1], kappa, xk, wk, K, LorY);

        PHI_K -= PHI_Kaux;
        PHI_V -= PHI_Vaux;
    }

}


void SA(REAL &PHI_K, REAL &PHI_V, REAL *y, REAL *x, REAL kappa, int same, 
        REAL K_diag, REAL V_diag, int LorY, REAL *xk, int xkSize, REAL *wk)
{
    // Put first panel at origin
    REAL y0_panel[3], y1_panel[3], y2_panel[3], x_panel[3];
    REAL X[3], Y[3], Z[3];
    for (int i=0; i<3;i++)
    {
        x_panel[i] = x[i] - y[i];
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
    intSide(PHI_K, PHI_V, panel0_final, panel1_final, x_plane[2], kappa, xk, wk, xkSize, LorY); // Side 0
    intSide(PHI_K, PHI_V, panel1_final, panel2_final, x_plane[2], kappa, xk, wk, xkSize, LorY); // Side 1
    intSide(PHI_K, PHI_V, panel2_final, panel0_final, x_plane[2], kappa, xk, wk, xkSize, LorY); // Side 2

    if (same==1)
    {
        PHI_K += K_diag;
        PHI_V += V_diag;
    }

}

void computeDiagonal(REAL *VL, int VLSize, REAL *KL, int KLSize, REAL *VY, int VYSize, REAL *KY, int KYSize, 
                    REAL *triangle, int triangleSize, REAL *centers, int centersSize, REAL kappa,
                    REAL K_diag, REAL V_diag, REAL *xk, int xkSize, REAL *wk, int wkSize)
{
    int N = VLSize, LorY;
    REAL PHI_K, PHI_V;
    for(int i=0; i<N; i++)
    {
        REAL panel[9] = {triangle[9*i], triangle[9*i+1], triangle[9*i+2],
                 triangle[9*i+3], triangle[9*i+4], triangle[9*i+5],
                 triangle[9*i+6], triangle[9*i+7], triangle[9*i+8]};
        REAL center[3] = {centers[3*i], centers[3*i+1], centers[3*i+2]};
    
        PHI_K = 0.;
        PHI_V = 0.;
        LorY = 1; // Laplace
        SA(PHI_K, PHI_V, panel, center, 1e-12, 1, 
            K_diag, V_diag, LorY, xk, xkSize, wk);

        VL[i] = PHI_V;
        KL[i] = PHI_K;

        PHI_K = 0.;
        PHI_V = 0.;
        LorY = 2; // Yukawa 
        SA(PHI_K, PHI_V, panel, center, kappa, 1, 
            K_diag, V_diag, LorY, xk, xkSize, wk);
        
        VY[i] = PHI_V;
        KY[i] = PHI_K;

    }
}

void GQ_fine(REAL &PHI_K, REAL &PHI_V, REAL *panel, REAL xi, REAL yi, REAL zi, 
            REAL kappa, REAL *Xk, REAL *Wk, int K_fine, REAL Area, int LorY)
{
    REAL nx, ny, nz;
    REAL dx, dy, dz, r, aux;

    PHI_K = 0.;
    PHI_V = 0.;

    aux = 1/(2*Area);
    nx = ((panel[4]-panel[1])*(panel[2]-panel[8]) - (panel[5]-panel[2])*(panel[1]-panel[7])) * aux;
    ny = ((panel[5]-panel[2])*(panel[0]-panel[6]) - (panel[3]-panel[0])*(panel[2]-panel[8])) * aux;
    nz = ((panel[3]-panel[0])*(panel[1]-panel[7]) - (panel[4]-panel[1])*(panel[0]-panel[6])) * aux;


    #pragma unroll
    for (int kk=0; kk<K_fine; kk++)
    {
        dx = xi - (panel[0]*Xk[3*kk] + panel[3]*Xk[3*kk+1] + panel[6]*Xk[3*kk+2]);
        dy = yi - (panel[1]*Xk[3*kk] + panel[4]*Xk[3*kk+1] + panel[7]*Xk[3*kk+2]);
        dz = zi - (panel[2]*Xk[3*kk] + panel[5]*Xk[3*kk+1] + panel[8]*Xk[3*kk+2]);
        r   = 1/sqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!

        if (LorY==1)
        {
            aux = Wk[kk]*Area*r;
            PHI_V += aux;
            PHI_K += aux*(nx*dx+ny*dy+nz*dz)*(r*r);
        }

        else
        {
            aux = Wk[kk]*Area*exp(-kappa*1/r)*r;
            PHI_V += aux;
            PHI_K += aux*(nx*dx+ny*dy+nz*dz)*r*(kappa+r);
        }

    }
}

void GQ_fineKt(REAL &PHI_Ktx, REAL &PHI_Kty, REAL &PHI_Ktz, REAL *panel, 
            REAL xi, REAL yi, REAL zi, REAL kappa, REAL *Xk, REAL *Wk, 
            int K_fine, REAL Area, int LorY)
{
    REAL dx, dy, dz, r, aux;

    PHI_Ktx = 0.;
    PHI_Kty = 0.;
    PHI_Ktz = 0.;

    #pragma unroll
    for (int kk=0; kk<K_fine; kk++)
    {
        dx = xi - (panel[0]*Xk[3*kk] + panel[3]*Xk[3*kk+1] + panel[6]*Xk[3*kk+2]);
        dy = yi - (panel[1]*Xk[3*kk] + panel[4]*Xk[3*kk+1] + panel[7]*Xk[3*kk+2]);
        dz = zi - (panel[2]*Xk[3*kk] + panel[5]*Xk[3*kk+1] + panel[8]*Xk[3*kk+2]);
        r   = 1/sqrt(dx*dx + dy*dy + dz*dz); // r is 1/r!!!

        if (LorY==1)
        {
            aux = Wk[kk]*Area*r*r*r;
            PHI_Ktx -= aux*dx;
            PHI_Kty -= aux*dy;
            PHI_Ktz -= aux*dz;
        }

        else
        {
            aux = Wk[kk]*Area*exp(-kappa*1/r)*r*r*(kappa+r);
            PHI_Ktx -= aux*dx;
            PHI_Kty -= aux*dy;
            PHI_Ktz -= aux*dz;
        }
    }
}

void direct_c(REAL *K_aux, int K_auxSize, REAL *V_aux, int V_auxSize, int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
        int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
        REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
        REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
        int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
        REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, 
        REAL kappa, REAL threshold, REAL eps, REAL w0, REAL *aux, int auxSize)
{
    double start,stop;
    int N_target = targetSize;
    int N_source = s_xjSize;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr;
    bool L_d, same, condition_an, condition_gq;

    for(int i_aux=0; i_aux<N_target; i_aux++)
    {  
        int i = target[i_aux];
        for(int j=0; j<N_source; j++)
        {   
            // Check if panels are far enough for Gauss quadrature
            dx_tri = xt[i_aux] - xi[tri[j]];
            dy_tri = yt[i_aux] - yi[tri[j]];
            dz_tri = zt[i_aux] - zi[tri[j]];
            R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
            
            L_d  = (sqrt(2*Area[tri[j]])/(R_tri+eps)>=threshold);
            same = (i==tri[j]);
            condition_an = ((same || L_d) && (k[j]==0));
            condition_gq = (!L_d);

            if(condition_gq)
            {
                //start = get_time();
                dx = xt[i_aux] - s_xj[j];
                dy = yt[i_aux] - s_yj[j];
                dz = zt[i_aux] - s_zj[j];
                R  = sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                R2 = R*R;
                R3 = R2*R;
                if (LorY==2)
                {
                    expKr = exp(-kappa*R);
                    V_aux[i_aux] += m[j]*expKr/R;
                    K_aux[i_aux] += expKr/R2*(kappa+1/R) * (dx*mx[j] + dy*my[j] + dz*mz[j]);
                }
                if (LorY==1)
                {
                    V_aux[i_aux] += m[j]/R;
                    K_aux[i_aux] += 1/R3*(dx*mx[j] + dy*my[j] + dz*mz[j]);
                }
                //stop = get_time();
                //aux[1] += stop - start;
            }
            
            if(condition_an)
            {
                aux[0] += 1;
                REAL center[3] = {xt[i_aux], yt[i_aux], zt[i_aux]};
                REAL panel[9]  = {triangle[9*tri[j]], triangle[9*tri[j]+1], triangle[9*tri[j]+2],
                                triangle[9*tri[j]+3], triangle[9*tri[j]+4], triangle[9*tri[j]+5],
                                triangle[9*tri[j]+6], triangle[9*tri[j]+7], triangle[9*tri[j]+8]};
                REAL PHI_K = 0., PHI_V = 0.;
                
                start = get_time();

                if (same==1)
                {
                    PHI_K = K_diag;
                    if (IorE==1)
                        PHI_V = sglInt_int[j];
                    else
                        PHI_V = sglInt_ext[j];
                }
                else
                {
                    GQ_fine(PHI_K, PHI_V, panel, xt[i_aux], yt[i_aux], zt[i_aux], kappa, Xsk, Wsk, WskSize, Area[tri[j]], LorY); 
                }


                stop = get_time();
                aux[1] += stop - start;

//                printf("%f \t %f\n",PHI_V,mVclean[j]);

                V_aux[i_aux]  += PHI_V * mVclean[j];
                K_aux[i_aux]  += PHI_K * mKclean[j]; 

            }
        }
    }

}

void direct_sort(REAL *K_aux, int K_auxSize, REAL *V_aux, int V_auxSize, int LorY, REAL K_diag, REAL V_diag, int IorE, REAL *triangle, int triangleSize,
        int *tri, int triSize, int *k, int kSize, REAL *xi, int xiSize, REAL *yi, int yiSize, 
        REAL *zi, int ziSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, 
        REAL *s_zj, int s_zjSize, REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        REAL *m, int mSize, REAL *mx, int mxSize, REAL *my, int mySize, REAL *mz, int mzSize, REAL *mKclean, int mKcleanSize, REAL *mVclean, int mVcleanSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, int *offSrc, int offSrcSize, int *offTwg, int offTwgSize,  
        int *target, int targetSize,REAL *Area, int AreaSize, REAL *sglInt_int, int sglInt_intSize, REAL *sglInt_ext, int sglInt_extSize, 
        REAL *xk, int xkSize, REAL *wk, int wkSize, REAL *Xsk, int XskSize, REAL *Wsk, int WskSize,
        REAL kappa, REAL threshold, REAL eps, REAL w0, REAL *aux, int auxSize)
{
    double start,stop;
    int CI_start, CI_end, CJ_start, CJ_end, list_start, list_end, CJ;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr, sum_K, sum_V;
    bool L_d, same, condition_an, condition_gq;

    for (int tarTwg=0; tarTwg<offTarSize; tarTwg++)
    {
        CI_start = offTar[tarTwg];
        CI_end   = offTar[tarTwg] + sizeTar[tarTwg];
        list_start = offTwg[tarTwg];
        list_end   = offTwg[tarTwg+1];

        for(int i=CI_start; i<CI_end; i++)
        {  
            sum_K = 0.;
            sum_V = 0.;

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = interList[lst];
                CJ_start = offSrc[CJ];
                CJ_end = offSrc[CJ+1];

                for(int j=CJ_start; j<CJ_end; j++)
                {   
                    // Check if panels are far enough for Gauss quadrature
                    //start = get_time();
                    int ptr = 9*j;
                    REAL panel[9]  = {triangle[ptr], triangle[ptr+1], triangle[ptr+2],
                                    triangle[ptr+3], triangle[ptr+4], triangle[ptr+5],
                                    triangle[ptr+6], triangle[ptr+7], triangle[ptr+8]};

                    dx_tri = xt[i] - (panel[0]+panel[3]+panel[6])/3;
                    dy_tri = yt[i] - (panel[1]+panel[4]+panel[7])/3;
                    dz_tri = zt[i] - (panel[2]+panel[5]+panel[8])/3;
                    R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
                    
                    L_d  = (sqrt(2*Area[j])/(R_tri+eps)>=threshold);
                    same = (R_tri<1e-12);
                    condition_an = ((L_d) && (k[j]==0));
                    condition_gq = (!L_d);
                    //stop = get_time();
                    //aux[1] += stop - start;

                    if(condition_gq)
                    {
                        //start = get_time();
                        dx = xt[i] - s_xj[j];
                        dy = yt[i] - s_yj[j];
                        dz = zt[i] - s_zj[j];
                        R  = sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                        R2 = R*R;
                        R3 = R2*R;
                        if (LorY==2)
                        {
                            expKr = exp(-kappa*R);
                            sum_V += m[j]*expKr/R;
                            sum_K += expKr/R2*(kappa+1/R) * (dx*mx[j] + dy*my[j] + dz*mz[j]);
                        }
                        if (LorY==1)
                        {
                            sum_V += m[j]/R;
                            sum_K += 1/R3*(dx*mx[j] + dy*my[j] + dz*mz[j]);
                        }
                        //stop = get_time();
                        //aux[1] += stop - start;
                    }
                    
                    if(condition_an)
                    {
                        start = get_time();
                        aux[0] += 1;
                        REAL center[3] = {xt[i], yt[i], zt[i]};
                        REAL PHI_K = 0., PHI_V = 0.;
                        
                        if (same==1)
                        {
                            PHI_K = K_diag;
                            if (IorE==1)
                                PHI_V = sglInt_int[j];
                            else
                                PHI_V = sglInt_ext[j];
                        }
                        else
                        {
                            GQ_fine(PHI_K, PHI_V, panel, xt[i], yt[i], zt[i], kappa, Xsk, Wsk, WskSize, Area[j], LorY); 
                        }

        //                printf("%f \t %f\n",PHI_V,mVclean[j]);

                        sum_V += PHI_V * mVclean[j];
                        sum_K += PHI_K * mKclean[j]; 
                        stop = get_time();
                        aux[1] += stop - start;

                    }
                }
            }

            V_aux[i] += sum_V;
            K_aux[i] += sum_K;
        }
    }
}

void directKt_sort(REAL *Ktx_aux, int Ktx_auxSize, REAL *Kty_aux, int Kty_auxSize, REAL *Ktz_aux, int Ktz_auxSize, 
        int LorY, REAL *triangle, int triangleSize,
        int *k, int kSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, REAL *s_zj, int s_zjSize, 
        REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        REAL *m, int mSize, REAL *mKclean, int mKcleanSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, 
        int *offSrc, int offSrcSize, int *offTwg, int offTwgSize, REAL *Area, int AreaSize,
        REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, REAL kappa, REAL threshold, REAL eps, REAL *aux, int auxSize)
{
    double start,stop;
    int CI_start, CI_end, CJ_start, CJ_end, list_start, list_end, CJ;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr, sum_Ktx, sum_Kty, sum_Ktz;
    bool L_d, same, condition_an, condition_gq;

    for (int tarTwg=0; tarTwg<offTarSize; tarTwg++)
    {
        CI_start = offTar[tarTwg];
        CI_end   = offTar[tarTwg] + sizeTar[tarTwg];
        list_start = offTwg[tarTwg];
        list_end   = offTwg[tarTwg+1];

        for(int i=CI_start; i<CI_end; i++)
        {  
            sum_Ktx = 0.;
            sum_Kty = 0.;
            sum_Ktz = 0.;

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = interList[lst];
                CJ_start = offSrc[CJ];
                CJ_end = offSrc[CJ+1];

                for(int j=CJ_start; j<CJ_end; j++)
                {   
                    // Check if panels are far enough for Gauss quadrature
                    //start = get_time();
                    int ptr = 9*j;
                    REAL panel[9]  = {triangle[ptr], triangle[ptr+1], triangle[ptr+2],
                                    triangle[ptr+3], triangle[ptr+4], triangle[ptr+5],
                                    triangle[ptr+6], triangle[ptr+7], triangle[ptr+8]};

                    dx_tri = xt[i] - (panel[0]+panel[3]+panel[6])/3;
                    dy_tri = yt[i] - (panel[1]+panel[4]+panel[7])/3;
                    dz_tri = zt[i] - (panel[2]+panel[5]+panel[8])/3;
                    R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
                    
                    L_d  = (sqrt(2*Area[j])/(R_tri+eps)>=threshold);
                    same = (R_tri<1e-12);
                    condition_an = ((L_d) && (k[j]==0));
                    condition_gq = (!L_d);
                    //stop = get_time();
                    //aux[1] += stop - start;

                    if(condition_gq)
                    {
                        //start = get_time();
                        dx = xt[i] - s_xj[j];
                        dy = yt[i] - s_yj[j];
                        dz = zt[i] - s_zj[j];
                        R  = sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                        R2 = R*R;
                        R3 = R2*R;
                        if (LorY==2)
                        {
                            expKr = m[j]*exp(-kappa*R)/R2*(kappa+1/R);
                            sum_Ktx -= expKr * dx;
                            sum_Kty -= expKr * dy;
                            sum_Ktz -= expKr * dz;
                        }
                        if (LorY==1)
                        {
                            expKr = m[j]/R3;
                            sum_Ktx -= expKr*dx;
                            sum_Kty -= expKr*dy;
                            sum_Ktz -= expKr*dz;
                        }
                        //stop = get_time();
                        //aux[1] += stop - start;
                    }
                    
                    if(condition_an)
                    {
                        start = get_time();
                        aux[0] += 1;
                        REAL PHI_Ktx = 0.;
                        REAL PHI_Kty = 0.;
                        REAL PHI_Ktz = 0.;
                        
                        if (same==1)
                        {
                            PHI_Ktx = 0;
                            PHI_Kty = 0;
                            PHI_Ktz = 0;
                        }
                        else
                        {
                            GQ_fineKt(PHI_Ktx, PHI_Kty, PHI_Ktz, panel, xt[i], yt[i], zt[i], kappa, Xsk, Wsk, WskSize, Area[j], LorY); 
                        }

        //                printf("%f \t %f\n",PHI_V,mVclean[j]);

                        sum_Ktx += PHI_Ktx * mKclean[j]; 
                        sum_Kty += PHI_Kty * mKclean[j]; 
                        sum_Ktz += PHI_Ktz * mKclean[j]; 
                        stop = get_time();
                        aux[1] += stop - start;

                    }
                }
            }

            Ktx_aux[i] += sum_Ktx;
            Kty_aux[i] += sum_Kty;
            Ktz_aux[i] += sum_Ktz;
        }
    }
}

void directKtqual_sort(REAL *Ktx_aux, int Ktx_auxSize, REAL *Kty_aux, int Kty_auxSize, REAL *Ktz_aux, int Ktz_auxSize, 
        int LorY, REAL *triangle, int triangleSize,
        int *k, int kSize, REAL *s_xj, int s_xjSize, REAL *s_yj, int s_yjSize, REAL *s_zj, int s_zjSize, 
        REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize,
        REAL *m, int mSize, REAL *mKclean, int mKcleanSize,
        int *interList, int interListSize, int *offTar, int offTarSize, int *sizeTar, int sizeTarSize, 
        int *offSrc, int offSrcSize, int *offTwg, int offTwgSize, REAL *Area, int AreaSize,
        REAL *Xsk, int XskSize, REAL *Wsk, int WskSize, REAL kappa, REAL threshold, REAL eps, REAL *aux, int auxSize)
{
    double start,stop;
    int CI_start, CI_end, CJ_start, CJ_end, list_start, list_end, CJ;
    REAL dx, dy, dz, dx_tri, dy_tri, dz_tri, R, R2, R3, R_tri, expKr, sum_Ktx, sum_Kty, sum_Ktz;
    bool L_d, same, condition_an, condition_gq;

    for (int tarTwg=0; tarTwg<offTarSize; tarTwg++)
    {
        CI_start = offTar[tarTwg];
        CI_end   = offTar[tarTwg] + sizeTar[tarTwg];
        list_start = offTwg[tarTwg];
        list_end   = offTwg[tarTwg+1];

        for(int i=CI_start; i<CI_end; i++)
        {  
            sum_Ktx = 0.;
            sum_Kty = 0.;
            sum_Ktz = 0.;

            int ptr = 9*i;
            REAL panel[9]  = {triangle[ptr], triangle[ptr+1], triangle[ptr+2],
                            triangle[ptr+3], triangle[ptr+4], triangle[ptr+5],
                            triangle[ptr+6], triangle[ptr+7], triangle[ptr+8]};

            for (int lst=list_start; lst<list_end; lst++)
            {
                CJ = interList[lst];
                CJ_start = offSrc[CJ];
                CJ_end = offSrc[CJ+1];

                for(int j=CJ_start; j<CJ_end; j++)
                {   
                    // Check if panels are far enough for Gauss quadrature
                    //start = get_time();

                    dx_tri = s_xj[j] - (panel[0]+panel[3]+panel[6])/3;
                    dy_tri = s_yj[j] - (panel[1]+panel[4]+panel[7])/3;
                    dz_tri = s_zj[j] - (panel[2]+panel[5]+panel[8])/3;
                    R_tri  = sqrt(dx_tri*dx_tri + dy_tri*dy_tri + dz_tri*dz_tri);
                    
                    L_d  = (sqrt(2*Area[i])/(R_tri+eps)>=threshold);
                    same = (R_tri<1e-12);
                    condition_an = ((L_d) && (k[i]==0));
                    condition_gq = (!L_d);
                    //stop = get_time();
                    //aux[1] += stop - start;

                    if(condition_gq)
                    {
                        //start = get_time();
                        dx = xt[i] - s_xj[j];
                        dy = yt[i] - s_yj[j];
                        dz = zt[i] - s_zj[j];
                        R  = sqrt(dx*dx + dy*dy + dz*dz + eps*eps);
                        R2 = R*R;
                        R3 = R2*R;
                        if (LorY==2)
                        {
                            expKr = m[j]*exp(-kappa*R)/R2*(kappa+1/R);
                            sum_Ktx -= expKr * dx;
                            sum_Kty -= expKr * dy;
                            sum_Ktz -= expKr * dz;
                        }
                        if (LorY==1)
                        {
                            expKr = m[j]/R3;
                            sum_Ktx -= expKr*dx;
                            sum_Kty -= expKr*dy;
                            sum_Ktz -= expKr*dz;
                        }
                        //stop = get_time();
                        //aux[1] += stop - start;
                    }
                    
                    if(condition_an)
                    {
                        start = get_time();
                        aux[0] += 1;
                        REAL PHI_Ktx = 0.;
                        REAL PHI_Kty = 0.;
                        REAL PHI_Ktz = 0.;
                        
                        if (same==1)
                        {
                            PHI_Ktx = 0;
                            PHI_Kty = 0;
                            PHI_Ktz = 0;
                        }
                        else
                        {
                            GQ_fineKt(PHI_Ktx, PHI_Kty, PHI_Ktz, panel, s_xj[j], s_yj[j], s_zj[j], kappa, Xsk, Wsk, WskSize, -1, LorY); // -1 because dx is flipped inside GQ_fineKt
                        }

        //                printf("%f \t %f\n",PHI_V,mVclean[j]);

                        sum_Ktx += PHI_Ktx * mKclean[j]; 
                        sum_Kty += PHI_Kty * mKclean[j]; 
                        sum_Ktz += PHI_Ktz * mKclean[j]; 
                        stop = get_time();
                        aux[1] += stop - start;

                    }
                }
            }

            Ktx_aux[i] += sum_Ktx;
            Kty_aux[i] += sum_Kty;
            Ktz_aux[i] += sum_Ktz;
        }
    }
}

void coulomb_direct(REAL *xt, int xtSize, REAL *yt, int ytSize, REAL *zt, int ztSize, 
                    REAL *m, int mSize, REAL *K_aux, int K_auxSize)
{
    REAL sum, dx, dy, dz, r;
    for(int i=0; i<xtSize; i++)
    {
        sum = 0.;
        for(int j=0; j<xtSize; j++)
        {
            if (i!=j)
            {
                dx = xt[i] - xt[j];
                dy = yt[i] - yt[j];
                dz = zt[i] - zt[j];
                r  = sqrt(dx*dx + dy*dy + dz*dz);
                sum += m[j]/r;
            }
        }
        K_aux[i] = m[i]*sum;
    }
}
