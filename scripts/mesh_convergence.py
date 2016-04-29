#!/usr/bin/env python

import os
import numpy
#from matplotlib import *
#from matplotlib.pyplot import *
#from matplotlib.backends.backend_pdf import PdfFile, PdfPages, FigureCanvasPdf
import sys
import math
sys.path.append('../util')

def scanOutput(filename):
    
    flag = 0
    for line in file(filename):
        line = line.split()
        if len(line)>0:
            if line[0]=='Converged':
                iterations = int(line[2])
            if line[0]=='Total' and line[1]=='elements':
                N = int(line[-1])
            if line[0]=='Totals:':
                flag = 1
            if line[0]=='Esolv' and flag==1:
                Esolv = float(line[2])
            if line[0]=='Esurf' and flag==1:
                Esurf = float(line[2])
            if line[0]=='Ecoul' and flag==1:
                Ecoul = float(line[2])
            if line[0]=='Time' and flag==1:
                Time = float(line[2])

    return N, iterations, Esolv, Esurf, Ecoul, Time
          
meshNumber = len(sys.argv) - 4
meshRefine = float(sys.argv[1])
paramFile = sys.argv[2]
inputFile = sys.argv[3]

mesh = []
for i in range(meshNumber):
    mesh.append(sys.argv[4+i])

comm = './main.py ' + paramFile + ' ' + inputFile 
out = 'regression_tests/output_aux'

print 'Start runs'
N = numpy.zeros(len(mesh))
iterations = numpy.zeros(len(mesh))
Esolv = numpy.zeros(len(mesh))
Esurf = numpy.zeros(len(mesh))
Ecoul = numpy.zeros(len(mesh))
Time = numpy.zeros(len(mesh))
for i in range(len(mesh)):
    print 'Start run for mesh '+mesh[i]
    cmd = comm + mesh[i] + '.config > ' + out
    os.system(cmd)
    print 'Scan output file'
    N[i],iterations[i],Esolv[i],Esurf[i],Ecoul[i],Time[i] = scanOutput(out)


# Richardson extrapolation
for i in range(meshNumber-2):
    print 'Using meshes ' + mesh[i] + ', ' + mesh[i+1] + ' and ' + mesh[i+2]
    q = (Esolv[i]*Esolv[i+2]-Esolv[i+1]*Esolv[i+1])/(Esolv[i]-2*Esolv[i+1]+Esolv[i+2])
    p = math.log((Esolv[i+2]-Esolv[i+1])/(Esolv[i+1]-Esolv[i]))/math.log(meshRefine)
    print 'q = %f, p = %f'%(q,p)


error = abs(Esolv-q)/abs(q)

print '\nNumber of elements : '+str(N)
print 'Number of iteration: '+str(iterations)
print 'Solvation energy   : '+str(Esolv)
print 'Richardson extrapol: %f kcal/mol'%q
print 'Observed order conv: %f'%p
print 'Error              : '+str(error)


