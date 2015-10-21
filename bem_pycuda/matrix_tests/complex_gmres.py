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

def gmres_mgs(A, M, x0, b, R, tol, max_iter, xtype=None):

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

    xtype = upcast(Atype, x.dtype, b.dtype)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation

    if iscomplexobj(zeros((1,), dtype=xtype)):
        [axpy, dotu, dotc, scal, rotg] =\
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal', 'rotg'], [x])
    else:
        # real type
        [axpy, dotu, dotc, scal, rotg] =\
            get_blas_funcs(['axpy', 'dot', 'dot',  'scal', 'rotg'], [x])

    # Make full use of direct access to BLAS by defining own norm
    def norm(z):
        return sqrt(real(dotc(z, z)))


    # Set number of outer and inner iterations
    max_outer = max_iter
    
    if R > dimen:
        warn('Setting number of inner iterations (restrt) to maximum\
              allowed, which is A.shape[0] ')
        R = dimen

    max_inner = R

    # Prep for method
    r = b - ravel(A*x)

    # Apply preconditioner
    r = ravel(M*r)
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



