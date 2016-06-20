"""
Generalized Minimum Residual Method (GMRES).

GMRES iteratively refines the initial solution guess to the system Ax=b. 

This implementation was based mainly on the gmres_mgs from PyAMG, where
modified Gram-Schmidt is used to orthogonalize the Krylov Space and
Givens Rotations are used to provide the residual norm each iteration.
 
https://github.com/pyamg/pyamg/blob/master/pyamg/krylov/_gmres_mgs.py

"""

