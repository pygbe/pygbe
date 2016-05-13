#!/usr/bin/env python
"""
It reads the .pqr file to generate the input for MSMS in order to
generate the mesh for the Stern layer.
"""
import sys
from numpy import *

file_in = sys.argv[1]
file_out = file_in+'.stern'

X = loadtxt(file_in)

X[:,3] += 2

savetxt(file_out, X, fmt='%5.5f')
