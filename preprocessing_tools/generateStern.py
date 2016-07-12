#!/usr/bin/env python
"""
It reads a .xyzr file that contains the positions of the atoms (x,y,z) and the
van der Waals radius (r), to generate the input for MSMS in order to generate
the mesh for the Stern layer.

file_in : filename.xyzr. Given as an argv when running the script.
file_out: filename.stern

Note: 
-----
You can create the .xyzr file from the .pqr, extracting the columns 6, 7, 8, 10
One possibility is type in the terminal:

awk '{print $6, $7, $8, $10}' filename.pqr > filename.xyzr
"""
import sys
import numpy

file_in = sys.argv[1]
file_out = file_in+'.stern'

X = numpy.loadtxt(file_in)

X[:,3] += 2

numpy.savetxt(file_out, X, fmt='%5.5f')
