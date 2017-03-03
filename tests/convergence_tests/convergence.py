import re
import os
import sys
import time
import numpy
import pickle
import datetime

try:
    import pycuda
except ImportError:
    ans = input('PyCUDA not found.  Regression tests will take forever.  Do you want to continue? [y/n] ')
    if ans in ['Y', 'y']:
        pass
    else:
        sys.exit()

from pygbe.main import main as pygbe

mesh = ['500', '2K', '8K', '32K']
lysozome_mesh = ['1','2','4','8']

def picklesave(test_outputs):
    with open('tests.pickle','wb') as f:
        pickle.dump(test_outputs, f, 2)

def pickleload():
    with open('tests.pickle', 'rb') as f:
        test_outputs = pickle.load(f)

    return test_outputs

def report_results(error, N, iterations, E, analytical, total_time, energy_type='Interaction', test_name=None):
    """
    Prints out information for the convergence tests.

    Inputs:
    -------
        error: list of float
            Error for each mesh case
        N: list of int
            Number of elements in test
        iterations: list of int
            Number of iterations to converge
        E: list of float
            Energy: either Total energy or Interaction energy
        analytical: list of float
            Interaction energy (analytical solution)
        total_time: list of float
            Total wall time of run i
        energy_type: str
            Label for energy (default 'Interaction')
    """
    with open('convergence_test_results', 'a') as f:
        print('-' * 60, file=f)
        print('{:-^60}'.format('Running: ' + test_name), file=f)
        print('-' * 60, file=f)
        print(datetime.datetime.now(), file=f)
        flag = 0
        for i in range(len(error)-1):
            rate = error[i]/error[i+1]
            if abs(rate-4)>0.6:
                flag = 1
                print('Bad convergence for mesh {} to {}, with rate {}'.
                      format(i, i+1, rate), file=f)

        if flag==0:
            print('Passed convergence test!', file=f)

        print('\nNumber of elements : {}'.format(N), file=f)
        print('Number of iteration: {}'.format(iterations), file=f)
        print('{} energy : {}'.format(energy_type, E), file=f)
        print('Analytical solution: {} kcal/mol'.format(analytical), file=f)
        print('Error              : {}'.format(error), file=f)
        print('Total time         : {}'.format(total_time), file=f)



def run_convergence(mesh, test_name, problem_folder, param, delete_output=True):
    """
    Runs convergence tests over a series of mesh sizes

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
    print('Runs for molecule + set phi/dphi surface')
    N = numpy.zeros(len(mesh))
    iterations = numpy.zeros(len(mesh))
    Esolv = numpy.zeros(len(mesh))
    Esurf = numpy.zeros(len(mesh))
    Ecoul = numpy.zeros(len(mesh))
    Time = numpy.zeros(len(mesh))
    for i in range(len(mesh)):
        try:
            print('Start run for mesh '+mesh[i])
            results = pygbe(['',
                            '-p', '{}'.format(param),
                            '-c', '{}_{}.config'.format(test_name, mesh[i]),
                            '-o', 'output_{}_{}'.format(test_name, mesh[i]),
                            '-g', './',
                            '{}'.format(problem_folder),], return_results_dict=True)

            N[i] = results['total_elements']
            iterations[i] = results['iterations']
            Esolv[i] = results.get('E_solv_kcal', 0)
            Esurf[i] = results.get('E_surf_kcal', 0)
            Ecoul[i] = results.get('E_coul_kcal', 0)
            Time[i] = results['total_time']

        except (pycuda._driver.MemoryError, pycuda._driver.LaunchError) as e:
            print('Mesh {} failed due to insufficient memory.'
                  'Skipping this test, but convergence test should still complete'.format(mesh[i]))
            time.sleep(4)


    return(N, iterations, Esolv, Esurf, Ecoul, Time)
