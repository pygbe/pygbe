#!/usr/bin/python
import sys
from numpy import *

file_in = sys.argv[1]
file_out = file_in+'.stern'

X = loadtxt(file_in)

X[:,3] += 2

savetxt(file_out, X, fmt='%5.3f')
