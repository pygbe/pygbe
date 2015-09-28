'''
  Copyright (C) 2013 by Christopher Cooper, Lorena Barba

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
'''

import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot
import sys
sys.path.append('../util')
from an_solution import *

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
            

tests = ['dirichlet_surface500.config','molecule_dirichlet_500.config','molecule_dirichlet_500_multi.config','molecule_dirichlet_cavityinside_500.config','molecule_dirichlet_inside_500.config','molecule_insideoutside.config','molecule_neumann_500.config','molecule_neumann_500_multi.config','molecule_neumann_cavityinside_500.config','molecule_neumann_inside_500.config','molecule_twoinside_500.config','molecule_twoinsideoutside_500.config','neumann_surface500.config']

comm = './main.py matrix_tests/input_files/sphere.param matrix_tests/input_files/'
out = 'matrix_tests/output'
comm_matrix = './matrix_tests/main_matrix.py matrix_tests/input_files/sphere.param matrix_tests/input_files/'
out_matrix = 'matrix_tests/output_matrix'

for i in range(len(tests)):
    print '\nStart run for test '+ tests[i]
    print 'PyGBe run'
    cmd = comm + tests[i] + ' > ' + out
    os.system(cmd)
    print 'Matrix run'
    cmd = comm_matrix + tests[i] + ' > ' + out_matrix
    os.system(cmd)

    print '\nScan output files'
    N,iterations,Esolv,Esurf,Ecoul,Time = scanOutput(out)
    N_m,iterations_m,Esolv_m,Esurf_m,Ecoul_m,Time_m = scanOutput(out_matrix)

    if iterations == iterations_m:
        print '\tSame number of iterations!'
    else:
        print '\tBAD: iterations differ by %i'%abs(iterations-iterations_m)
    e = abs(Esolv-Esolv_m)/abs(Esolv+1e-16)
    if e<1e-6:
        print '\tSolvation energy matches!'
    else:
        print '\tBAD: solvation energy differs by %i%%'%(e*100)

    e = abs(Esurf-Esurf_m)/abs(Esurf+1e-16)
    if e<1e-6:
        print '\tSurface energy matches!'
    else:
        print '\tBAD: surface energy differs by %i%%'%(e*100)

    e = abs(Ecoul-Ecoul_m)/abs(Ecoul+1e-16)
    if e<1e-6:
        print '\tCoulomb energy matches!'
    else:
        print '\tBAD: coulomb energy differs by %i%%'%(e*100)

    print '\nCheck intermediate steps'

    thres = 1e-12
    print 'Test RHS (threshold = %s)'%thres
    a   = loadtxt('RHS.txt')
    a_m = loadtxt('RHS_matrix.txt')
    e = abs(a-a_m)/abs(a_m+1e-16)
    if max(e)<thres:
        print '\tRHS matches!'
    else:
        print '\tBAD: RHS differs by %s'%max(e)

    thres = 1e-7
    print 'Test GMRES Vip1 (threshold = %s)'%thres
    for it in range(1,6):
        f  = 'Vip1'+str(it)+'.txt'
        fm = 'Vip1'+str(it)+'_matrix.txt'
        a   = loadtxt(f)
        a_m = loadtxt(fm)
        e = abs(a-a_m)/abs(a_m+1e-16)
        if max(e)<thres:
            print '\tVip1 of iteration %i matches!'%it
        else:
            print '\tBAD: Vip1 of iteration %i differs by %s'%(it,max(e))

    thres = 1e-4
    print 'Test GMRES result (threshold = %s)'%thres
    a   = loadtxt('phi.txt')
    a_m = loadtxt('phi_matrix.txt')
    e = abs(a-a_m)/abs(a_m+1e-16)
    if max(e)<thres:
        print '\tGMRES result matches!'
    else:
        print '\tBAD: GMRES result differs by %s'%max(e)
