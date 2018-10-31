#include <math.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#define REAL double

void calc_aux_cy(REAL *q , int qSize, 
                    REAL *xq0 , int xq0Size,
                    REAL *xq1 , int xq1Size,
                    REAL *xq2 , int xq2Size,
                    REAL *xi , int xiSize,
                    REAL *yi , int yiSize,
                    REAL *zi , int ziSize,
                    REAL *normal0 , int normal0Size, 
                    REAL *normal1 , int normal1Size, 
                    REAL *normal2 , int normal2Size,
                    int stype,
                    REAL *aux , int auxSize,
                    REAL E);

void calc_aux_cy(REAL *q , int qSize, 
                    REAL *xq0 , int xq0Size,
                    REAL *xq1 , int xq1Size,
                    REAL *xq2 , int xq2Size,
                    REAL *xi , int xiSize,
                    REAL *yi , int yiSize,
                    REAL *zi , int ziSize,
                    REAL *normal0 , int normal0Size, 
                    REAL *normal1 , int normal1Size, 
                    REAL *normal2 , int normal2Size,
                    int stype,
                    REAL *aux , int auxSize,
                    REAL E)
{
    #pragma omp parallel default(none) shared(qSize, xiSize, xi, yi, zi, xq0, xq1, xq2, auxSize, aux, q, normal0, normal1, normal2, E, stype)
    {
    REAL auxiliar;

        #pragma omp for nowait
        for (int j = 0; j < xiSize; j++)
        {
            for (int i=0; i<qSize; i++)
            {
                if (stype == 1)
                {
                    auxiliar = - ( q[i] / ( ( sqrt( (xi[j] - xq0[i]) * (xi[j] - xq0[i]) + (yi[j] - xq1[i]) * (yi[j] - xq1[i]) + (zi[j] - xq2[i]) * (zi[j] - xq2[i]) ) ) * ( sqrt( (xi[j] - xq0[i]) * (xi[j] - xq0[i]) + (yi[j] - xq1[i]) * (yi[j] - xq1[i]) + (zi[j] - xq2[i]) * (zi[j] - xq2[i]) ) ) * ( sqrt( (xi[j] - xq0[i]) * (xi[j] - xq0[i]) + (yi[j] - xq1[i]) * (yi[j] - xq1[i]) + (zi[j] - xq2[i]) * (zi[j] - xq2[i]) ) ) ) * ( (xi[j] - xq0[i]) * normal0[j] + (yi[j] - xq1[i]) * normal1[j] + (zi[j] - xq2[i]) * normal2[j] ) );
                } else
                {
                    auxiliar = q[i] / (E * ( sqrt( (xi[j] - xq0[i]) * (xi[j] - xq0[i]) + (yi[j] - xq1[i]) * (yi[j] - xq1[i]) + (zi[j] - xq2[i]) * (zi[j] - xq2[i]) ) )) ;
                }

                aux[j] = aux[j] + auxiliar;
            }
        }
    }
};

