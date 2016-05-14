import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pickle

from pygbe.util import an_solution
from regression import (scanOutput, run_regression, picklesave, pickleload,
                        report_results, mesh)


def main():
    print('{:-^60}'.format('Running two_molecules test'))
    try:
        test_outputs = pickleload()
    except IOError:
        test_outputs = {}

    problem_folder = 'input_files'

    #twosphere
    print('Runs for two molecules')
    param = 'sphere_fine.param'
    test_name = 'twosphere'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name,
            problem_folder,
            param, delete_output=False)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    #molecule_single_center
    print('Runs for isolated molecule')
    param = 'sphere_fine.param'
    test_name = 'molecule_single_center'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name,
            problem_folder,
            param, delete_output=False)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    #load results for analysis
    Esolv, Esurf, Ecoul = test_outputs['twosphere'][2:5]
    Esolv_single, Esurf_single, Ecoul_single = test_outputs[
        'molecule_single_center'][2:5]
    Time = test_outputs['twosphere'][-1]
    Time_single = test_outputs['molecule_single_center'][-1]
    N, iterations = test_outputs['twosphere'][:2]

    Etotal = Esolv + Esurf + Ecoul
    total_time = Time
    Esolv_single *= 2  # Same molecule twice

    Einter = Esolv + Esurf + Ecoul - Esurf_single - Ecoul_single - Esolv_single
    total_time = Time + Time_single

    analytical, EE1, EE2 = an_solution.two_sphere(5., 12., 0.125, 4., 80., 1.)
    analytical *= 2

    error = abs(Einter - analytical) / abs(analytical)

    report_results(error, N, iterations, Einter, analytical, total_time)

#    font = {'family':'serif','size':10}
#    fig = plt.figure(figsize=(3,2), dpi=80)
#    ax = fig.add_subplot(111)
#    asymp = N[0]*error[0]/N
#    ax.loglog(N, error, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
#    ax.loglog(N, asymp, c='k', marker='None', ls=':', lw=0.8, label=None)
#    plt.rc('font',**font)
#    loc = (3*N[0]+N[1])/4
#    tex_loc = array((loc,N[0]*error[0]/loc))
#    tex_angle = math.atan2(numpy.log(abs(asymp[-1]-asymp[0])),numpy.log(abs(N[-1]-N[0])))*180/math.pi
#    ax.text(tex_loc[0], tex_loc[1],r'N$^{-1}$',fontsize=8,rotation=tex_angle,rotation_mode='anchor')
#    ax.set_ylabel('Relative error', fontsize=10)
#    ax.set_xlabel('Number of elements', fontsize=10)
#    fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
#    fig.savefig('regression_tests/figs/error_energy_twomolecule.pdf',dpi=80,format='pdf')
#
#    fig = plt.figure(figsize=(3,2), dpi=80)
#    ax = fig.add_subplot(111)
#    asymp = N*numpy.log(N)*total_time[0]/(N[0]*numpy.log(N[0]))
#    ax.loglog(N, total_time, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
#    ax.loglog(N, asymp,c='k',marker='None',ls=':', lw=0.8, label=None)
#    loc = (3*N[0]+N[1])/4
#    tex_loc = array((loc, loc*numpy.log(loc)*total_time[0]/(N[0]*numpy.log(N[0]))))
#    tex_angle = math.atan2(numpy.log(abs(asymp[-1]-asymp[0])),numpy.log(abs(N[-1]-N[0])))*180/math.pi
#    ax.text(tex_loc[0], tex_loc[1], 'NlogN', fontsize=8,rotation=tex_angle, rotation_mode='anchor')
#    plt.rc('font',**font)
#    ax.set_ylabel('Total time [s]', fontsize=10)
#    ax.set_xlabel('Number of elements', fontsize=10)
#    fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
#    fig.savefig('regression_tests/figs/total_time_twomolecule.pdf',dpi=80,format='pdf')
#
#    fig = plt.figure(figsize=(3,2), dpi=80)
#    ax = fig.add_subplot(111)
#    ax.semilogx(N, iterations, c='k', marker='o',ls=' ', mfc='w', ms=5, label='')
#    plt.rc('font',**font)
#    ax.set_ylabel('Iterations', fontsize=10)
#    ax.set_xlabel('Number of elements', fontsize=10)
#    fig.subplots_adjust(left=0.185, bottom=0.21, right=0.965, top=0.95)
#    fig.savefig('regression_tests/figs/iterations_twomolecule.pdf',dpi=80,format='pdf')
#

if __name__ == "__main__":
    main()
