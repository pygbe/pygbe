#include <stdio.h>
#include <time.h>
#include <cmath>
#include <cstdlib>
#define REAL double

void direct_c(REAL *xi, int xiSize, 
              REAL *yi, int yiSize, 
              REAL *zi, int ziSize, 
              REAL *xj, int xjSize, 
              REAL *yj, int yjSize, 
              REAL *zj, int zjSize, 
              REAL *ds, int dsSize, 
              REAL *dsx, int dsxSize, 
              REAL *dsy, int dsySize, 
              REAL *dsz, int dszSize, 
              REAL *m, int mSize, 
              REAL kappa, REAL eps)
{   
    /*
    double kappa = 1.5;

    double eps = 1e-10;
    double *xi = new double[N];
    double *yi = new double[N];
    double *zi = new double[N];
    double *xj = new double[N];
    double *yj = new double[N];
    double *zj = new double[N];
    double *m = new double[N];
    double *ds = new double[N];
    double *dsx = new double[N];
    double *dsy = new double[N];
    double *dsz = new double[N];

    for (int i; i<N; i++)
    {
        xi[i] = rand()/(1.+RAND_MAX);
        yi[i] = rand()/(1.+RAND_MAX);
        zi[i] = rand()/(1.+RAND_MAX);
        xj[i] = rand()/(1.+RAND_MAX);
        yj[i] = rand()/(1.+RAND_MAX);
        zj[i] = rand()/(1.+RAND_MAX);
        m[i] = 1.0f/N;
    }   
    clock_t begin = clock();
    */

    // Direct summation
    REAL r, r2, dx, dy, dz; 
    //time_t begin, end;
    //time (&begin);
    for (int i=0; i<xiSize; i++)
    {   
        //ds[i] = -m[i]/sqrt(eps);
        for (int j=0; j<xjSize; j++)
        {   
            dx = xi[i] - xj[j]; 
            dy = yi[i] - yj[j]; 
            dz = zi[i] - zj[j]; 
            r = sqrt(dx*dx + dy*dy + dz*dz + eps);
            r2 = r*r;
            ds[i] += m[j]*exp(-kappa*r)/r;
            dsx[i] += -m[j]*dx*exp(-kappa*r)/r2*(kappa+1/r);
            dsy[i] += -m[j]*dy*exp(-kappa*r)/r2*(kappa+1/r);
            dsz[i] += -m[j]*dz*exp(-kappa*r)/r2*(kappa+1/r);

        }   
    }   
    /*
    clock_t end = clock();
    //time (&end);
    //double time_sec = difftime(end,begin);

    double time_sec = double(end-begin)/CLOCKS_PER_SEC;
//    time_sec /= 100;

    printf ("Time: %fs\n",time_sec);
    return 0;
    */
}
