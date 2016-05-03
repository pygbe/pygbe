import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy
import re
import sys
import math
from pygbe.util import an_solution
from pygbe.main import main as pygbe

ITER_REGEX = re.compile('Converged after (\d*) iterations')
N_REGEX = re.compile('Total elements : (\d*)')
ESOLV_REGEX = re.compile('Esolv = (\-*\d*\.\d*)\ kcal\/mol')
ESURF_REGEX = re.compile('Esurf = (\-*\d*\.\d*)\ kcal\/mol')
ECOUL_REGEX = re.compile('Ecoul = (\-*\d*\.\d*)\ kcal\/mol')
TIME_REGEX = re.compile('Time = (\-*\d*\.\d*)\ s')

def scanOutput(filename):

    with open(filename, 'r') as f:
        txt = f.read()

        N= re.search(N_REGEX, txt)
        if N:
            N = int(N.group(1))
        iterations = re.search(ITER_REGEX, txt)
        if iterations:
            iterations = int(iterations.group(1))
        Esolv = re.search(ESOLV_REGEX, txt)
        if Esolv:
            Esolv = float(Esolv.group(1))
        Esurf = re.search(ESURF_REGEX, txt)
        if Esurf:
            Esurf = float(Esurf.group(1))
        Ecoul = re.search(ECOUL_REGEX, txt)
        if Ecoul:
            Ecoul = float(Ecoul.group(1))
        Time = re.search(TIME_REGEX, txt)
        if Time:
            Time = float(Time.group(1))


    return N, iterations, Esolv, Esurf, Ecoul, Time



def run_regression(mesh, problem_folder, param):
    """
    Runs regression tests over a series of mesh sizes

    Inputs:
    ------
        mesh: array of mesh suffixes
        problem_folder: str name of folder containing meshes, etc...
        param: str name of param file

    Returns:
    -------
        N: len(mesh) array of elements of problem
        iterations: # of iterations to converge
        Esolv: array of solvation energy
        Esurf: array of surface energy
        Ecoul: array of coulomb energy
        Time: time to solution (wall-time)
    """
    print 'Runs for molecule + set phi/dphi surface'
    N = numpy.zeros(len(mesh))
    iterations = numpy.zeros(len(mesh))
    Esolv = numpy.zeros(len(mesh))
    Esurf = numpy.zeros(len(mesh))
    Ecoul = numpy.zeros(len(mesh))
    Time = numpy.zeros(len(mesh))
    for i in range(len(mesh)):
        print 'Start run for mesh '+mesh[i]
        outfile = pygbe(['',
                         '-p', '{}'.format(param),
                         '-c', '{}_{}.config'.format(problem_folder, mesh[i]),
                         '-o', 'output_{}'.format(mesh[i]),
                         '-g', '../../pygbe/',
                         '{}'.format(problem_folder),], return_output_fname=True)

        print 'Scan output file'
        outfile = os.path.join('{}'.format(problem_folder),
                            'output_{}'.format(mesh[i]),
                            outfile)
        N[i],iterations[i],Esolv[i],Esurf[i],Ecoul[i],Time[i] = scanOutput(outfile)


    return(N, iterations, Esolv, Esurf, Ecoul, Time)


def main():
    #molecule_dirichlet
    mesh = ['500','2K','8K','32K','130K']
    param = 'sphere_fine.param'
    problem_folder = 'molecule_dirichlet'
    N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(mesh, problem_folder, param)

    #molecule_single_center
    mesh = ['500','2K','8K','32K','130K']
    param = 'sphere_fine.param'
    problem_folder = 'molecule_single_center'
    N_mol, iterations_mol, Esolv_mol, Esurf_mol, Ecoul_mol, Time_mol = run_regression(mesh, problem_folder, param)

    #dirichlet_surface
    mesh = ['500','2K','8K','32K','130K']
    param = 'sphere_fine.param'
    problem_folder = 'dirichlet_surface'
    N_surf, iterations_surf, Esolv_surf, Esurf_surf, Ecoul_surf, Time_surf = run_regression(mesh, problem_folder, param)


    Einter = Esolv + Esurf + Ecoul - Esolv_surf - Esurf_mol - Ecoul_mol - Esolv_mol - Esurf_surf - Ecoul_surf
    total_time = Time+Time_mol+Time_surf

    analytical = an_solution.molecule_constant_potential(1., 1., 5., 4., 12., 0.125, 4., 80.)  

    error = abs(Einter-analytical)/abs(analytical)

    print '\nNumber of elements : '+str(N)
    print 'Number of iteration: '+str(iterations)
    print 'Interaction energy : '+str(Einter)
    print 'Analytical solution: %f kcal/mol'%analytical
    print 'Error              : '+str(error)
    print 'Total time         : '+str(total_time)
#
#
#flag = 0
#for i in range(len(error)-1):
#    rate = error[i]/error[i+1]
#    if abs(rate-4)>0.6:
#        flag = 1
#        print 'Bad convergence for mesh %i to %i, with rate %f'%(i,i+1,rate)
#
#if flag==0:
#    print '\nPassed convergence test!'
#
#
#font = {'family':'serif','size':10}
#fig = plt.figure(figsize=(3,2), dpi=80)
#ax = fig.add_subplot(111)
#asymp = N[0]*error[0]/N
#ax.loglog(N, error, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
#ax.loglog(N, asymp, c='k', marker='None', ls=':', lw=0.8, label=None)
#plt.rc('font',**font)
#loc = (3*N[0]+N[1])/4
#tex_loc = array((loc,N[0]*error[0]/loc))
#tex_angle = math.atan2(numpy.log(abs(asymp[-1]-asymp[0])),numpy.log(abs(N[-1]-N[0])))*180/math.pi
#ax.text(tex_loc[0], tex_loc[1],r'N$^{-1}$',fontsize=8,rotation=tex_angle,rotation_mode='anchor')
#ax.set_ylabel('Relative error', fontsize=10)
#ax.set_xlabel('Number of elements', fontsize=10)
#fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
#fig.savefig('regression_tests/figs/error_energy_molecule_dirichlet.pdf',dpi=80,format='pdf')
#
#fig = plt.figure(figsize=(3,2), dpi=80)
#ax = fig.add_subplot(111)
#asymp = N*numpy.log(N)*total_time[0]/(N[0]*numpy.log(N[0]))
#ax.loglog(N, total_time, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
#ax.loglog(N, asymp,c='k',marker='None',ls=':', lw=0.8, label=None)
#loc = (3*N[0]+N[1])/4
#tex_loc = array((loc, loc*numpy.log(loc)*total_time[0]/(N[0]*numpy.log(N[0]))))
#tex_angle = math.atan2(numpy.log(abs(asymp[-1]-asymp[0])),numpy.log(abs(N[-1]-N[0])))*180/math.pi
#ax.text(tex_loc[0], tex_loc[1], 'NlogN', fontsize=8,rotation=tex_angle, rotation_mode='anchor')
#plt.rc('font',**font)
#ax.set_ylabel('Total time [s]', fontsize=10)
#ax.set_xlabel('Number of elements', fontsize=10)
#fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
#fig.savefig('regression_tests/figs/total_time_molecule_dirichlet.pdf',dpi=80,format='pdf')
#
#fig = plt.figure(figsize=(3,2), dpi=80)
#ax = fig.add_subplot(111)
#ax.semilogx(N, iterations, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
#plt.rc('font',**font)
#ax.set_ylabel('Iterations', fontsize=10)
#ax.set_xlabel('Number of elements', fontsize=10)
#fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
#fig.savefig('regression_tests/figs/iterations_molecule_dirichlet.pdf',dpi=80,format='pdf')

if __name__ == "__main__":
    main()
