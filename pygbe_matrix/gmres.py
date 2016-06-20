"""
Generalized Minimum Residual Method (GMRES).

GMRES iteratively refines the initial solution guess to the system Ax=b. 

This implementation was based mainly on the gmres_mgs from PyAMG, where
modified Gram-Schmidt is used to orthogonalize the Krylov Space and
Givens Rotations are used to provide the residual norm each iteration.
 
https://github.com/pyamg/pyamg/blob/master/pyamg/krylov/_gmres_mgs.py

"""

import numpy 
import scipy
from scipy.linalg import get_blas_funcs, solve
from scipy.sparse.sputils import upcast
from scipy.sparse.linalg import gmres as scipy_gmres
from warnings import warn
import time

#Defining the function to calculate the Givens rotations

def apply_givens(Q, v, k):
    """
    Apply the first k Givens rotations in Q to the vector v.

    Parameter
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


