#Following gmres_mgs from pyamg

# For robustness, modified Gram-Schmidt is used to orthogonalize the Krylov
# Space Givens Rotations are used to provide the residual norm each iteration

import numpy 
import scipy
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
from scipy.linalg import get_blas_funcs
from warnings import warn

#Defining the function to calculate the Givens rotations

def apply_givens(Q, v, k):
    '''
    Apply the first k Givens rotations in Q to the vector v

    Parameter
    ---------        
        Q: {list}, list of consecutive 2x2 Givens rotations
        v: {array}, vector to apply the rotations to
        k: {int}, number of rotations to apply

    Returns
    -------
    v that is changed in place

    '''

    for j in xrange(k):
        Qloc = Q[j]
        v[j:j+2] = scipy.dot(Qloc, v[j:j+2])

def gmres_mgs(A, x0, b, R, tol, max_iter, M=None, xtype=None):

    '''
    Generalized Minimum Residual Method (GMRES)
        GMRES iteratively refines the initial solution guess to the system
        Ax = b
        Modified Gram-Schmidt version
    
    Parameter
    ---------        
        A: {matrix, Linear Operator}, n x n, linear system to solve.
        M: {matrix, LinearOperator}, n x n, preconditioner.
        x: {array, matrix}, initial guess.        
        b: {array, matrix}, right hand side, (n, ) or (n,1)
        R: {int}, number of iterations for GMRES to do restart.
        tol: {float}, relative convergence tolerance.
        max_iter: {int}, maximum number of GMRES iterations.
        xtype : {type}, dtype for the solution, default is automatic type detection
        
    Returns
    -------
    xNew : an updated guess to the solution of Ax = b   
    '''
    # Convert inputs to linear system, with error checking
    A, M, x, b, postprocess = make_system(A, M, x0, b, xtype)
    dimen = A.shape[0]    

    if not hasattr(A, 'dtype'):
        Atype = upcast(x.dtype, b.dtype)
    else:
        Atype = A.dtype
    if not hasattr(M, 'dtype'):
        Mtype = upcast(x.dtype, b.dtype)
    else:
        Mtype = M.dtype

    xtype = upcast(Atype, x.dtype, b.dtype, Mtype)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation

    if numpy.iscomplexobj(numpy.zeros((1,), dtype=xtype)):
        [axpy, dotu, dotc, scal, rotg] =\
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal', 'rotg'], [x])
    else:
        # real type
        [axpy, dotu, dotc, scal, rotg] =\
            get_blas_funcs(['axpy', 'dot', 'dot',  'scal', 'rotg'], [x])

    # Make full use of direct access to BLAS by defining own norm
    def norm(z):
        return numpy.sqrt(numpy.real(dotc(z, z)))


    # Set number of outer and inner iterations
    max_outer = max_iter
    
    if R > dimen:
        warn('Setting number of inner iterations (restrt) to maximum\
              allowed, which is A.shape[0] ')
        R = dimen

    max_inner = R

    # Prep for method
    r = b - numpy.ravel(A*x)

    # Apply preconditioner
    r = numpy.ravel(M*r)
    normr = norm(r)
    
    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol*normb:
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_2, we use the preconditioned residual
    # because this is left preconditioned GMRES.
    if normr != 0.0:
        tol = tol*normr

    iteration = 0
    
    #Here start the GMRES
    for outer in xrange(max_outer):

        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).
        # NOTE:  We are dealing with row-major matrices, so we traverse in a
        #        row-major fashion,
        #        i.e., H and V's transpose is what we store.
        
        Q = []  # Initialzing Givens Rotations
        # Upper Hessenberg matrix, which is then
        # converted to upper triagonal with Givens Rotations

        H = numpy.zeros((max_inner+1, max_inner+1), dtype=xtype)
        V = numpy.zeros((max_inner+1, dimen), dtype=xtype) #Krylov space

        # vs store the pointers to each column of V.
        #   This saves a considerable amount of time.
        vs = []

        # v = r/normr
        V[0, :] = scal(1.0/normr, r) # scal wrapper of dscal --> x = a*x  
        vs.append(V[0, :])

        #RHS vector in the Krylov space
        g = numpy.zeros((dimen, ), dtype=xtype)
        g[0] = normr

        for inner in xrange(max_inner):
            #New search direction
            v= V[inner+1, :]
            v[:] = numpy.ravel(M*(A*vs[-1]))
            vs.append(v)
            normv_old = norm(v)

            #Modified Gram Schmidt
            for k in xrange(inner+1):                
                vk = vs[k]
                alpha = dotc(vk, v)
                H[inner, k] = alpha
                v[:] = axpy(vk, v, dimen, -alpha)  # y := a*x + y
                # axpy is a wrapper for daxpy, daxpy(x,y,[n,a,offx,incx,offy,incy])
                # n: input int, that specifies the number of elements in x and y (dimen)
                # a: Specifies the scalar (-alpha)
    
            normv = norm(v)
            H[inner, inner+1] = normv

#Here the pymag check for re-orthogonalize not sure if necesasary.

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
                    #rotg is a blas function that computes the parameters
                    #for a Givens rotation
                    [c, s] = rotg(H[inner, inner], H[inner, inner+1])
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
                
                if iteration%1==0:
                    print ('Iteration: %i, residual: %s'%(iteration,normr))                

                if normr < tol:
                    break

        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space V.  Solve inner x inner system.
        y = scipy.linalg.solve (H[0:inner+1, 0:inner+1].T, g[0:inner+1])
        update = numpy.ravel(scipy.mat(V[:inner+1, :]).T * y.reshape(-1,1))
        x= x + update            
        r = b - numpy.ravel(A*x)

        # Apply preconditioner
        r = numpy.ravel(M*r)
        normr = norm(r)

        if normr < tol:
            return (postprocess(x), 0) #0 means everything went well!

     # end outer loop

    return (postprocess(x), iteration)

#Testing 

from scipy.linalg import solve
from scipy.sparse.linalg import gmres as scipy_gmres
import time

xmin = -1.
xmax = 1.
N = 5000
h = (xmax-xmin)/(N-1)
x = numpy.arange(xmin, xmax+h/2, h)

A = numpy.zeros((N,N))
for i in range(N):
    A[i] = numpy.exp(-abs(x-x[i])**2/(2*h**2))

b = numpy.random.random(N)
x = numpy.zeros(N)
R = 50
max_iter = 5000
tol = 1e-8

tic = time.time()
(xg, flag) = gmres_mgs(A, x, b, R, tol, max_iter) #Not preconditioner
toc = time.time()
print 'Time for my GMRES_mgs: %fs'%(toc-tic)


tic = time.time()
xs = solve(A, b)
toc = time.time()
print 'Time for straight solve: %fs'%(toc-tic)

tic = time.time()
xsg = scipy_gmres(A, b, x, tol, R, max_iter)[0]
toc = time.time()
print 'Time for scipy GMRES: %fs'%(toc-tic)

error_xs_xg = numpy.sqrt(sum((xs-xg)**2)/sum(xs**2))
print 'error stright solve vs mgs_gmres: %s'%error_xs_xg

error_xs_xsg = numpy.sqrt(sum((xs-xsg)**2)/sum(xs**2))
print 'error stright solve vs scipy_gmres: %s'%error_xs_xsg



