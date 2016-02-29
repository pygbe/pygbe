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
from numpy import zeros, array
import math
import sys

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
            

mesh = array(['1','2','4','8'])

comm = 'pygbe regression_tests/input_files/lys.param regression_tests/input_files/lys_single_'
out = 'regression_tests/output_aux'

N = zeros(len(mesh))

print 'Simulations for Lysozyme with single surface'
iterations_single = zeros(len(mesh))
Esolv_single = zeros(len(mesh))
Esurf_single = zeros(len(mesh))
Ecoul_single = zeros(len(mesh))
Time_single = zeros(len(mesh))
for i in range(len(mesh)):
    print 'Start run for mesh '+mesh[i]
    cmd = comm + mesh[i] + '.config > ' + out
    os.system(cmd)
    print 'Scan output file'
    N[i],iterations_single[i],Esolv_single[i],Esurf_single[i],Ecoul_single[i],Time_single[i] = scanOutput(out)


comm = 'pygbe regression_tests/input_files/lys.param regression_tests/input_files/lys_'
print 'Simulations for Lysozyme full surface simulation'
iterations_full = zeros(len(mesh))
Esolv_full = zeros(len(mesh))
Esurf_full = zeros(len(mesh))
Ecoul_full = zeros(len(mesh))
Time_full = zeros(len(mesh))
for i in range(len(mesh)):
    print 'Start run for mesh '+mesh[i]
    cmd = comm + mesh[i] + '.config > ' + out
    os.system(cmd)
    print 'Scan output file'
    N[i],iterations_full[i],Esolv_full[i],Esurf_full[i],Ecoul_full[i],Time_full[i] = scanOutput(out)

comm = 'pygbe regression_tests/input_files/lys.param regression_tests/input_files/lys_k0_'
print 'Simulations for Lysozyme with kappa=0'
iterations_k0 = zeros(len(mesh))
Esolv_k0 = zeros(len(mesh))
Esurf_k0 = zeros(len(mesh))
Ecoul_k0 = zeros(len(mesh))
Time_k0 = zeros(len(mesh))
for i in range(len(mesh)):
    print 'Start run for mesh '+mesh[i]
    cmd = comm + mesh[i] + '.config > ' + out
    os.system(cmd)
    print 'Scan output file'
    N[i],iterations_k0[i],Esolv_k0[i],Esurf_k0[i],Ecoul_k0[i],Time_k0[i] = scanOutput(out)


Esolv_ref_single = 1/4.184*array([-2401.2, -2161.8, -2089, -2065.5])   
Esolv_ref_full = 1/4.184*array([-2432.9, -2195.9, -2124.2, -2101.1])
Esolv_FFTSVD = array([-577.105, -520.53, -504.13, -498.26])# Remember FFTSVD was only run with kappa=0

iter_ref_single = array([33,34,35,39])
iter_ref_full = array([36,38,41,45])
iter_FFTSVD = array([32,34,35,37])

error_single = abs(Esolv_single-Esolv_ref_single)/abs(Esolv_ref_single)
error_full   = abs(Esolv_full-Esolv_ref_full)/abs(Esolv_ref_full)
error_FFTSVD = abs(Esolv_k0-Esolv_FFTSVD)/abs(Esolv_FFTSVD)

iter_diff_single = iterations_single - iter_ref_single
iter_diff_full   = iterations_full - iter_ref_full
iter_diff_FFTSVD = iterations_k0 - iter_FFTSVD


flag = 0
thresh = 1e-2
for i in range(len(error_single)):
    if error_single[i]>thresh:
        flag = 1
        print 'Solvation energy not agreeing for single surface simulation, mesh %i by %f'%(i,error_single[i])

    if error_full[i]>thresh:
        flag = 1
        print 'Solvation energy not agreeing for full surface simulation, mesh %i by %f'%(i,error_full[i])

    if error_FFTSVD[i]>thresh:
        flag = 1
        print 'Solvation energy not agreeing with FFTSVD, mesh %i by %f'%(i,error_FFTSVD[i])

if flag==0:
    print '\nPassed Esolv test!'
else:
    print '\nFAILED Esolv test'

flag = 0
thresh = 3
for i in range(len(iter_diff_single)):
    if abs(iter_diff_single[i])>thresh:
        flag = 1
        print 'Solvation energy not agreeing for single surface simulation, mesh %i by %f'%(i,iter_diff_single[i])

    if abs(iter_diff_full[i])>thresh:
        flag = 1
        print 'Solvation energy not agreeing for full surface simulation, mesh %i by %f'%(i,iter_diff_full[i])

    if abs(iter_diff_FFTSVD[i])>thresh:
        flag = 1
        print 'Solvation energy not agreeing with FFTSVD, mesh %i by %f'%(i,iter_diff_FFTSVD[i])

if flag==0:
    print '\nPassed iterations test! They are all within %i iterations of reference'%thresh
else:
    print '\nFAILED iterations test'

print 'Summary:'
print 'Single: Esolv: '+str(Esolv_single)+', iterations: '+str(iterations_single)
print 'Full  : Esolv: '+str(Esolv_full)+', iterations: '+str(iterations_full)
print 'k=0   : Esolv: '+str(Esolv_k0)+', iterations: '+str(iterations_k0)
