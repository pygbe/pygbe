import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from numpy import zeros, array
import math
import sys
from pygbe.util import an_solution

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
            

mesh = array(['500','2K','8K','32K','130K'])

comm = 'pygbe regression_tests/input_files/sphere_fine.param regression_tests/input_files/molecule_stern_'
out = 'regression_tests/output_aux'

N = zeros(len(mesh))
iterations = zeros(len(mesh))
Esolv = zeros(len(mesh))
Esurf = zeros(len(mesh))
Ecoul = zeros(len(mesh))
Time = zeros(len(mesh))
for i in range(len(mesh)):
    print 'Start run for mesh '+mesh[i]
    cmd = comm + mesh[i] + '.config > ' + out
    os.system(cmd)
    print 'Scan output file'
    N[i],iterations[i],Esolv[i],Esurf[i],Ecoul[i],Time[i] = scanOutput(out)

total_time = Time

analytical = an_solution.an_P(array([1.]), array([[1.,1.,1.41421356]]), 4., 80., 4., 0.125, 5., 20)  

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
tex_angle = math.atan2(numpy.log(abs(asymp[-1]-asymp[0])),numpy.log(abs(N[-1]-N[0])))*180/math.pi
ax.text(tex_loc[0], tex_loc[1],r'N$^{-1}$',fontsize=8,rotation=tex_angle,rotation_mode='anchor')
ax.set_ylabel('Relative error', fontsize=10)
ax.set_xlabel('Number of elements', fontsize=10)
fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
fig.savefig('regression_tests/figs/error_energy_sphere_molecule_stern.pdf',dpi=80,format='pdf')

fig = plt.figure(figsize=(3,2), dpi=80)
ax = fig.add_subplot(111)
asymp = N*numpy.log(N)*total_time[0]/(N[0]*numpy.log(N[0]))
ax.loglog(N, total_time, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
ax.loglog(N, asymp,c='k',marker='None',ls=':', lw=0.8, label=None)
loc = (3*N[0]+N[1])/4
tex_loc = array((loc, loc*numpy.log(loc)*total_time[0]/(N[0]*numpy.log(N[0]))))
tex_angle = math.atan2(numpy.log(abs(asymp[-1]-asymp[0])),numpy.log(abs(N[-1]-N[0])))*180/math.pi
ax.text(tex_loc[0], tex_loc[1], 'NlogN', fontsize=8,rotation=tex_angle, rotation_mode='anchor')
plt.rc('font',**font)
ax.set_ylabel('Total time [s]', fontsize=10)
ax.set_xlabel('Number of elements', fontsize=10)
fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
fig.savefig('regression_tests/figs/total_time_sphere_molecule_stern.pdf',dpi=80,format='pdf')

fig = plt.figure(figsize=(3,2), dpi=80)
ax = fig.add_subplot(111)
ax.semilogx(N, iterations, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
plt.rc('font',**font)
ax.set_ylabel('Iterations', fontsize=10)
ax.set_xlabel('Number of elements', fontsize=10)
fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
fig.savefig('regression_tests/figs/iterations_sphere_molecule_stern.pdf',dpi=80,format='pdf')
