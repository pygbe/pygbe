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


def picklesave(test_outputs):
    with open('tests','wb') as f:
        pickle.dump(test_outputs, f, 2)

def pickleload():
    with open('tests', 'rb') as f:
        test_outputs = pickle.load(f)

    return test_outputs

def mesh_ratio(N):
    """
    Calculates the mesh refinement ratio between consecutive meshes.
    
    Arguments:
    ----------
    N: list of int, Number of elements in test.

    Returns:
    --------
    mesh_ratio: list of float, mesh refinement ratio between consequtive meshes.         
    """ 
    mesh_ratio = []
    for i in range(len(N)-1):
        mesh_ratio.append(N[i+1]/N[i])

    return mesh_ratio


def report_results(error, N, expected_rate, iterations, Cext, analytical, total_time, test_name=None):
    """
    Prints out information for the convergence tests.

    Inputs:
    -------
        error        : list of float, L2 Norm of error against analytical solution.
        N            : list of int, Number of elements in test.
        expected_rate: float, expected error rate acording to mesh refinement. 
        iterations   : list of int, Number of iterations to converge.
        Cext         : list of float, Cross extinction section.
        analytical   : list of float, analytical solution of the Cross extinction 
                                   section.
        total_time: list of float, total wall time of run i.
    """
    with open('convergence_test_results', 'a') as f:
        print('-' * 60, file=f)
        print('{:-^60}'.format('Running: ' + test_name), file=f)
        print('-' * 60, file=f)
        print(datetime.datetime.now(), file=f)
        flag = 0
        for i in range(len(error)-1):
            rate = error[i]/error[i+1]
            if abs(rate-expected_rate)>0.6:
                flag = 1
                print('Bad convergence for mesh {} to {}, with rate {}'.
                      format(i, i+1, rate), file=f)

        if flag==0:
            print('Passed convergence test!', file=f)

        print('\nNumber of elements : {}'.format(N), file=f)
        print('Number of iteration: {}'.format(iterations), file=f)
        print('Cext'.format(Cext), file=f)
        print('Analytical solution: {} kcal/mol'.format(analytical), file=f)
        print('Error              : {}'.format(error), file=f)
        print('Total time         : {}'.format(total_time), file=f)


def run_convergence(mesh, test_name, problem_folder, param):
    """
    Runs convergence tests over a series of mesh sizes

    Inputs:
    ------
        mesh          : array of mesh suffixes
        problem_folder: str, name of folder containing meshes, etc...
        param         : str, name of param file

    Returns:
    -------
        N         : len(mesh) array, elements of problem.
        iterations: len(mesh) array, number of iterations to converge.
        Cext      : len(mesh) array of float, Cross extinction section.
        Time      : len(mesh) array of float, time to solution (wall-time)
    """
    print('Runs lspr case of silver sphere in water medium')
    N = numpy.zeros(len(mesh))
    iterations = numpy.zeros(len(mesh))
    Cext = numpy.zeros(len(mesh))
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
            Cext[i] = results.get('Cext', 0)
            Time[i] = results['total_time']

            mesh_ratio = mesh_ratio(N)
            expected_rate = 0

            if all(ratio==mesh_ratio[0] for ratio in mesh_ratio):
                expected_rate = mesh_ratio[0]
            else:                
                print('Mesh ratio inconsistency. \nCheck that the mesh ratio' 
                      'remains constant along refinement'
                      'Convergence test report will bad convergence for this reason')
                 

        except (pycuda._driver.MemoryError, pycuda._driver.LaunchError) as e:
            print('Mesh {} failed due to insufficient memory.'
                  'Skipping this test, but convergence test should still complete'.format(mesh[i]))
            time.sleep(4)


    return(N, iterations, expected_rate, Cext, Time)
