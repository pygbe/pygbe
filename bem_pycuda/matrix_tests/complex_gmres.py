#Following gmres_mgs from pyamg

# For robustness, modified Gram-Schmidt is used to orthogonalize the Krylov
# Space Givens Rotations are used to provide the residual norm each iteration

import numpy 
import scipy
from scipy.sparse.sputils import upcast
from scipy.linalg import get_blas_funcs

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
