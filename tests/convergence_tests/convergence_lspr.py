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
