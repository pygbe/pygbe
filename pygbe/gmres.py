"""
Generalized Minimum Residual Method (GMRES).

GMRES iteratively refines the initial solution guess to the system Ax=b.

This implementation was based mainly on the gmres_mgs from PyAMG, where
modified Gram-Schmidt is used to orthogonalize the Krylov Space and
Givens Rotations are used to provide the residual norm each iteration.

Guidance code:

 - PyAMG library:
      https://github.com/pyamg/pyamg/blob/master/pyamg/krylov/_gmres_mgs.py
"""

import numpy
import scipy
from warnings import warn

from scipy.linalg import get_blas_funcs, get_lapack_funcs
from scipy.sparse.sputils import upcast

from pygbe.matrixfree import gmres_dot


def apply_givens(Q, v, k):
    """
    Apply the first k Givens rotations in Q to the vector v.

    Arguments
    ---------
        Q: list, list of consecutive 2x2 Givens rotations
        v: array, vector to apply the rotations to
        k: int, number of rotations to apply

    Returns
    -------
        v: array, that is changed in place.

    """

    for j in range(k):
        Qloc = Q[j]
        v[j:j+2] = scipy.dot(Qloc, v[j:j+2])


def gmres_mgs(surf_array, field_array, X, b, param, ind0, timing, kernel):
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
    iteration  : int, number of outer iterations for convergence

    References
    ----------
    .. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
       Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003
       http://www-users.cs.umn.edu/~saad/books.html
    .. [2] C. T. Kelley, http://www4.ncsu.edu/~ctk/matlab_roots.html
    """

    # Defining xtype as dtype of the problem, to decide which BLAS functions
    # import.
    xtype = upcast(X.dtype, b.dtype)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation

    [lartg] = get_lapack_funcs(['lartg'], [X] )
    if numpy.iscomplexobj(numpy.zeros((1,), dtype=xtype)):
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [X])
    else:
        # real type
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [X])

    # Make full use of direct access to BLAS by defining own norm
    def norm(z):
        return numpy.sqrt(numpy.real(dotc(z, z)))

    # Defining dimension
    dimen = len(X)

    max_iter = param.max_iter
    R = param.restart
    tol = param.tol

    # Set number of outer and inner iterations
    
    if R > dimen:
        warn('Setting number of inner iterations (restrt) to maximum\
              allowed, which is A.shape[0] ')
        R = dimen

    max_inner = R

    #max_outer should be max_iter/max_inner but this might not be an integer
    #so we get the ceil of the division.
    #In the inner loop there is a if statement to break in case max_iter is
    #reached. 
 
    max_outer = int(numpy.ceil(max_iter/max_inner))

    # Prep for method
    aux = gmres_dot(X, surf_array, field_array, ind0, param, timing, kernel)
    r = b - aux

    normr = norm(r)

    # Check initial guess ( scaling by b, if b != 0, must account for
    # case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        return X

    iteration = 0

    # Here start the GMRES
    for outer in range(max_outer):

        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).
        # NOTE:  We are dealing with row-major matrices, so we traverse in a
        #        row-major fashion,
        #        i.e., H and V's transpose is what we store.

        Q = []  # Initialzing Givens Rotations
        # Upper Hessenberg matrix, which is then
        # converted to upper triagonal with Givens Rotations

        H = numpy.zeros((max_inner+1, max_inner+1), dtype=xtype)
        V = numpy.zeros((max_inner+1, dimen), dtype=xtype)  # Krylov space

        # vs store the pointers to each column of V.
        # This saves a considerable amount of time.
        vs = []

        # v = r/normr
        V[0, :] = scal(1.0/normr, r)  # scal wrapper of dscal --> x = a*x
        vs.append(V[0, :])

        #Saving initial residual to be used to calculate the rel_resid
        if iteration==0:
            res_0 = normb

        #RHS vector in the Krylov space
        g = numpy.zeros((dimen, ), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):
            #New search direction
            v= V[inner+1, :]
            v[:] = gmres_dot(vs[-1], surf_array, field_array, ind0, param,
 timing, kernel)
            vs.append(v)

            #Modified Gram Schmidt
            for k in range(inner+1):
                vk = vs[k]
                alpha = dotc(vk, v)
                H[inner, k] = alpha
                v[:] = axpy(vk, v, dimen, -alpha)  # y := a*x + y
                #axpy is a wrapper for daxpy (blas function)

            normv = norm(v)
            H[inner, inner+1] = normv


            #Check for breakdown
            if H[inner, inner+1] != 0.0:
                v[:] = scal(1.0/H[inner, inner+1], v)

            #Apply for Givens rotations to H
            if inner > 0:
                apply_givens(Q, H[inner, :], inner)

            #Calculate and apply next complex-valued Givens rotations

            #If max_inner = dimen, we don't need to calculate, this
            #is unnecessary for the last inner iteration when inner = dimen -1

            if inner != dimen - 1:
                if H[inner, inner+1] != 0:
                    #lartg is a lapack function that computes the parameters
                    #for a Givens rotation
                    [c, s, _] = lartg(H[inner, inner], H[inner, inner+1])
                    Qblock = numpy.array([[c, s], [-numpy.conjugate(s),c]], dtype=xtype)
                    Q.append(Qblock)

                    #Apply Givens Rotations to RHS for the linear system in
                    # the krylov space.
                    g[inner:inner+2] = scipy.dot(Qblock, g[inner:inner+2])

                    #Apply Givens rotations to H
                    H[inner, inner] = dotu(Qblock[0,:], H[inner, inner:inner+2])
                    H[inner, inner+1] = 0.0

            iteration+= 1

            if inner < max_inner-1:
                normr = abs(g[inner+1])
                rel_resid = normr/res_0

                if rel_resid < tol:
                    break

            if iteration%1==0:
                print('Iteration: {}, relative residual: {}'.format(iteration,rel_resid))

            if (inner + 1 == R):
                print('Residual: {}. Restart...'.format(rel_resid))

            if iteration==max_iter:
                print('Warning!!!!'
                'You have reached the maximum number of iterations : {}.'.format(iteration))
                print('The run will stop. Check the residual behaviour you might have a bug.'
                'For future runs you might consider changing the tolerance or'
                ' increasing the number of max_iter.')

                break     
                
        # end inner loop, back to outer loop

        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y = scipy.linalg.solve (H[0:inner+1, 0:inner+1].T, g[0:inner+1])
        update = numpy.ravel(scipy.mat(V[:inner+1, :]).T * y.reshape(-1,1))
        X= X + update
        aux = gmres_dot(X, surf_array, field_array, ind0, param, timing, kernel)
        r = b - aux

        normr = norm(r)
        rel_resid = normr/res_0

        # test for convergence
        if rel_resid < tol:
            print('GMRES solve')
            print('Converged after {} iterations to a residual of {}'.format(iteration,rel_resid))
            print('Time weight vector: {}'.format(timing.time_mass))
            print('Time sort         : {}'.format(timing.time_sort))
            print('Time data transfer: {}'.format(timing.time_trans))
            print('Time P2M          : {}'.format(timing.time_P2M))
            print('Time M2M          : {}'.format(timing.time_M2M))
            print('Time M2P          : {}'.format(timing.time_M2P))
            print('Time P2P          : {}'.format(timing.time_P2P))
            print('\tTime analy: {}'.format(timing.time_an))

            return X, iteration

    #end outer loop

    return X, iteration
