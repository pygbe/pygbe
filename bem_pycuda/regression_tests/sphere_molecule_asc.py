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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from numpy import zeros, array
import math
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
            

mesh = array(['500','2K','8K','32K'])#,'130K'])

comm = './main.py regression_tests/input_files/sphere_fine.param regression_tests/input_files/molecule_single_asc_'
out = 'regression_tests/output_aux'

N = zeros(len(mesh))
iterations = zeros(len(mesh))
Esolv = zeros(len(mesh))
Esurf = zeros(len(mesh))
Ecoul = zeros(len(mesh))
Time = zeros(len(mesh))
for i in range(len(mesh)):
    print 'Start run for mesh '+mesh[i]
    cmd = comm + mesh[i] + '.config qualocation > ' + out
    os.system(cmd)
    print 'Scan output file'
    N[i],iterations[i],Esolv[i],Esurf[i],Ecoul[i],Time[i] = scanOutput(out)

total_time = Time

analytical = an_P(array([1.]), array([[1.,1.,1.41421356]]), 4., 80., 5., 0., 5., 30)  

error = abs(Esolv-analytical)/abs(analytical)

flag = 0
for i in range(len(error)-1):
    rate = error[i]/error[i+1]
    if abs(rate-4)>0.6:
        flag = 1
        print 'Bad convergence for mesh %i to %i, with rate %f'%(i,i+1,rate)

if flag==0:
    print '\nPassed convergence test!'

print '\nNumber of elements  : '+str(N)
print 'Number of iterations: '+str(iterations)
print 'Total energy        : '+str(Esolv)
print 'Analytical solution : %f kcal/mol'%analytical
print 'Error               : '+str(error)
print 'Total time          : '+str(total_time)

font = {'family':'serif','size':10}
fig = plt.figure(figsize=(3,2), dpi=80)
ax = fig.add_subplot(111)
asymp = N[0]*error[0]/N
ax.loglog(N, error, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
ax.loglog(N, asymp, c='k', marker='None', ls=':', lw=0.8, label=None)
plt.rc('font',**font)
loc = (3*N[0]+N[1])/4
tex_loc = array((loc,N[0]*error[0]/loc))
tex_angle = math.atan2(log(abs(asymp[-1]-asymp[0])),log(abs(N[-1]-N[0])))*180/math.pi
ax.text(tex_loc[0], tex_loc[1],r'N$^{-1}$',fontsize=8,rotation=tex_angle,rotation_mode='anchor')
ax.set_ylabel('Relative error', fontsize=10)
ax.set_xlabel('Number of elements', fontsize=10)
fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
fig.savefig('regression_tests/figs/error_energy_sphere_asc.pdf',dpi=80,format='pdf')

fig = plt.figure(figsize=(3,2), dpi=80)
ax = fig.add_subplot(111)
asymp = N*log(N)*total_time[0]/(N[0]*log(N[0]))
ax.loglog(N, total_time, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
ax.loglog(N, asymp,c='k',marker='None',ls=':', lw=0.8, label=None)
loc = (3*N[0]+N[1])/4
tex_loc = array((loc, loc*log(loc)*total_time[0]/(N[0]*log(N[0]))))
tex_angle = math.atan2(log(abs(asymp[-1]-asymp[0])),log(abs(N[-1]-N[0])))*180/math.pi
ax.text(tex_loc[0], tex_loc[1], 'NlogN', fontsize=8,rotation=tex_angle, rotation_mode='anchor')
plt.rc('font',**font)
ax.set_ylabel('Total time [s]', fontsize=10)
ax.set_xlabel('Number of elements', fontsize=10)
fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
fig.savefig('regression_tests/figs/total_time_sphere_asc.pdf',dpi=80,format='pdf')

fig = plt.figure(figsize=(3,2), dpi=80)
ax = fig.add_subplot(111)
ax.semilogx(N, iterations, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
plt.rc('font',**font)
ax.set_ylabel('Iterations', fontsize=10)
ax.set_xlabel('Number of elements', fontsize=10)
fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
fig.savefig('regression_tests/figs/iterations_sphere_asc.pdf',dpi=80,format='pdf')
