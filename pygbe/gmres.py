"""
Generalized Minimum Residual Method (GMRES).

This implementation is a RESTARTED version of the algorithm and uses:

 - Arnoldi-Modified Gram-Schmidt method to reduced the dense matrix into an
   upper Hessenberg matrix. 
 - Givens Rotations to reduce the Hessenberg matrix to an upper tringular, and
   to solve the least squares problem.

References:

 - For Arnoldi-Modified Gram-Schmidt:
    Iterative methods for sparse linear systems - Yousef Saad - 2nd ed. (2000).
    (pg. 148).
 - For Givens Rotations implementation:
    Iterative methods for linear and non-linear equations - C.T Kelley - (1995).         
    (pg. 43-45).
 - For RESTART version:
    Saad's book (pg. 167)

Guidance code:

 - CUSP library:
    https://github.com/cusplibrary/cusplibrary/blob/develop/cusp/krylov/
    detail/gmres.inl   
"""

import numpy
import os
from scipy import linalg
import time
from matrixfree import gmres_dot as gmres_dot


def GeneratePlaneRotation(dx, dy, cs, sn):
    """
    Given a vector (dx, dy), it provides the cosine (cs) and sine (sn). 
  
    Arguments
    ----------
    dx: float, x coordinate of the vector (dx, dy). 
    dy: float, y coordinate of the vector (dx, dy).
    cs: float, cosine.
    sn: float, sine.
        
    Returns
    --------
    cs: float, cosine.
    sn: float, sine. 
    """

    if dy == 0:
        cs = 1.
        sn = 0.
    elif (abs(dy) > abs(dx)):
        temp = dx / dy
        sn = 1 / numpy.sqrt(1 + temp * temp)
        cs = temp * sn
    else:
        temp = dy / dx
        cs = 1 / numpy.sqrt(1 + temp * temp)
        sn = temp * cs

    return cs, sn


def ApplyPlaneRotation(dx, dy, cs, sn):
    """
    Given a vector (dx, dy), the cosine (cs) and sine (sn), it rotates the
    vector.  

    Arguments
    ----------
    dx: float, x coordinate of the vector (dx, dy). 
    dy: float, y coordinate of the vector (dx, dy).
    cs: float, cosine.
    sn: float, sine. 
        
    Returns
    --------
    dx: float, x component of the rotated vector.
    dy: float, y component of the rotated vector. 
    """
    
    temp = cs * dx + sn * dy
    dy = -sn * dx + cs * dy
    dx = temp

    return dx, dy


def PlaneRotation(H, cs, sn, s, i, R):
    """
    It applies the Givens rotations. 

    Arguments
    ----------
    H : matrix, to applied the rotations. In our case is the upper Hessenberg
                obtained after Arnoldi-Modified Gram-Schmidt. 
    cs: float, cosine.
    sn: float, sine.
    s : array, residual.
    i : int, iteration number of the GMRES.
    R : int, restart parameter, number of iterations for GMRES to do restart.
    
    Returns
    --------
    H : 
    cs: float, cosine.
    sn: float, sine.
    s : array, residual.
    """

    for k in range(i):
        H[k, i], H[k + 1, i] = ApplyPlaneRotation(H[k, i], H[k + 1, i], cs[k],
                                                  sn[k])

    cs[i], sn[i] = GeneratePlaneRotation(H[i, i], H[i + 1, i], cs[i], sn[i])
    H[i, i], H[i + 1, i] = ApplyPlaneRotation(H[i, i], H[i + 1, i], cs[i],
                                              sn[i])
    s[i], s[i + 1] = ApplyPlaneRotation(s[i], s[i + 1], cs[i], sn[i])

    return H, cs, sn, s


def gmres_solver(surf_array, field_array, X, b, param, ind0, timing, kernel):
    """
    GMRES solver. 

    Arguments
    ----------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    field_array: array, contains the Field classes of each region on the surface.
    X          : array, initial guess. 
    b          : array, right hand side.
    param      : class, parameters related to the surface.     
    ind0       : class, it contains the indices related to the treecode 
                        computation.     
    timing     : class, it contains timing information for different parts of 
                        the code.
    kernel     : pycuda source module.

    Returns
    --------
    X          : array, an updated guess to the solution. 
    """

    N = len(b)
    V = numpy.zeros((param.restart + 1, N))
    H = numpy.zeros((param.restart + 1, param.restart))

    time_Vi = 0.
    time_Vk = 0.
    time_rotation = 0.
    time_lu = 0.
    time_update = 0.

    # Initializing varibles
    rel_resid = 1.
    cs, sn = numpy.zeros(N), numpy.zeros(N)

    iteration = 0

    b_norm = linalg.norm(b)

    output_path = os.path.join(
        os.environ.get('PYGBE_PROBLEM_FOLDER'), 'OUTPUT')

    while (iteration < param.max_iter and
           rel_resid >= param.tol):  # Outer iteration

        aux = gmres_dot(X, surf_array, field_array, ind0, param, timing,
                        kernel)

        r = b - aux
        beta = linalg.norm(r)

        if iteration == 0:
            print 'Analytical integrals: %i of %i, %i' % (
                timing.AI_int / param.N, param.N, 100 * timing.AI_int / param.N
                **2) + '%'

        V[0, :] = r[:] / beta
        if iteration == 0:
            res_0 = b_norm

        s = numpy.zeros(param.restart + 1)
        s[0] = beta
        i = -1

        while (i + 1 < param.restart and
               iteration + 1 <= param.max_iter):  # Inner iteration
            i += 1
            iteration += 1

            # Compute Vip1
            tic = time.time()

            Vip1 = gmres_dot(V[i, :], surf_array, field_array, ind0, param,
                             timing, kernel)
            toc = time.time()
            time_Vi += toc - tic

            if iteration < 6:
                fname = 'Vip1{}.txt'.format(iteration)
#                numpy.savetxt(os.path.join(output_path,fname), Vip1)

            tic = time.time()
            Vk = V[0:i + 1, :]
            H[0:i + 1, i] = numpy.dot(Vip1, numpy.transpose(Vk))

            # This ends up being slower than looping           
            #            HVk = H[0:i+1,i]*numpy.transpose(Vk)
            #            Vip1 -= HVk.numpy.sum(axis=1)

            for k in range(i + 1):
                Vip1 -= H[k, i] * Vk[k]
            toc = time.time()
            time_Vk += toc - tic

            H[i + 1, i] = linalg.norm(Vip1)
            V[i + 1, :] = Vip1[:] / H[i + 1, i]

            tic = time.time()
            H, cs, sn, s = PlaneRotation(H, cs, sn, s, i, param.restart)
            toc = time.time()
            time_rotation += toc - tic

            rel_resid = abs(s[i + 1]) / res_0

            if iteration % 1 == 0:
                print 'iteration: %i, rel resid: %s' % (iteration, rel_resid)

            if (i + 1 == param.restart):
                print('Residual: %f. Restart...' % rel_resid)
            if rel_resid <= param.tol:
                break

        # Solve the triangular system
        tic = time.time()
        piv = numpy.arange(i + 1)
        y = linalg.lu_solve((H[0:i + 1, 0:i + 1], piv), s[0:i + 1], trans=0)
        toc = time.time()
        time_lu += toc - tic

        # Update solution
        tic = time.time()
        Vj = numpy.zeros(N)
        for j in range(i + 1):
            # Compute Vj
            Vj[:] = V[j, :]
            X += y[j] * Vj
        toc = time.time()
        time_update += toc - tic

#    print 'Time Vip1    : %fs'%time_Vi
#    print 'Time Vk      : %fs'%time_Vk
#    print 'Time rotation: %fs'%time_rotation
#    print 'Time lu      : %fs'%time_lu
#    print 'Time update  : %fs'%time_update
    print 'GMRES solve'
    print 'Converged after %i iterations to a residual of %s' % (iteration,
                                                                 rel_resid)
    print 'Time weight vector: %f' % timing.time_mass
    print 'Time sort         : %f' % timing.time_sort
    print 'Time data transfer: %f' % timing.time_trans
    print 'Time P2M          : %f' % timing.time_P2M
    print 'Time M2M          : %f' % timing.time_M2M
    print 'Time M2P          : %f' % timing.time_M2P
    print 'Time P2P          : %f' % timing.time_P2P
    print '\tTime analy: %f' % timing.time_an
    #    print 'Tolerance: %f, maximum iterations: %f'%(tol, max_iter)

    return X

