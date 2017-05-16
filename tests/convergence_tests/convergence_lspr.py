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

from pygbe.lspr import main as pygbe

mesh = ['500', '2K', '8K', '32K']
mesh_multiple = ['500-100', '2K-500', '8K-2K', '32K-8K']


def picklesave(test_outputs):
    with open('tests.pickle','wb') as f:
        pickle.dump(test_outputs, f, 2)

def pickleload():
    with open('tests.pickle', 'rb') as f:
        test_outputs = pickle.load(f)

    return test_outputs

def mesh_ratio(N):
    """
    Calculates the mesh refinement ratio between consecutive meshes.
    
    Arguments:
    ----------
    N: list, Number of elements / avg_density in test (depends on what applies).

    Returns:
    --------
    mesh_ratio: list of float, mesh refinement ratio between consequtive meshes.         
    """ 
    mesh_ratio = []
    for i in range(len(N)-1):
        mesh_ratio.append(N[i+1]/N[i])

    return mesh_ratio


def report_results(error, N, expected_rate, iterations, Cext_0, total_time, analytical=None,  rich_extra=None, avg_density=None, test_name=None):
    """
    Prints out information for the convergence tests.

    Inputs:
    -------
        error        : list of float, error of the calculation per mesh case.
        N            : list , Number of elements in test.
        expected_rate: float, expected error rate acording to mesh refinement. 
        iterations   : list of int, Number of iterations to converge.
        Cext_0       : float, Cross extinction section of the main sphere.
        total_time   : list of float, total wall time of run i.
        analytical   : float, analytical solution of the Cross extinction 
                                   section (when applicable).
        rich_extra   : float, richardson extrapolation solution of the Cross
                              extinction section (when applicable).
        avg_density  : list, avegerage density per mesh, N_total/total_Area 
                            (when applicable).
        test_name    : str, name you want to assign to the test in report.      
    """
    with open('convergence_test_results', 'a') as f:
        print('-' * 60, file=f)
        print('{:-^60}'.format('Running: ' + test_name), file=f)
        print('-' * 60, file=f)
        print(datetime.datetime.now(), file=f)
        flag = 0
        for i in range(len(error)-1):
            rate = error[i]/error[i+1]
            if abs(rate-expected_rate)>0.4:
                flag = 1
                print('Bad convergence for mesh {} to {}, with rate {}'.
                      format(i, i+1, rate), file=f)

        if flag==0:
            print('Passed convergence test!', file=f)

        print('\nNumber of elements : {}'.format(N), file=f)
        if avg_density:
             print('Average density, elem/nm^2 : {}'.format(avg_density), file=f)
        print('Number of iteration: {}'.format(iterations), file=f)
        print('Cross extinction section surface 0 nm^2: {}'.format(Cext_0), file=f)
        if analytical:        
            print('Analytical solution: {} nm^2'.format(analytical), file=f)
        if rich_extra:
            print('Richardson extrapolation solution: {} nm^2'.format(rich_extra), file=f)        
        print('Error              : {}'.format(error), file=f)
        print('Total time         : {}'.format(total_time), file=f)


def run_convergence(mesh, test_name, problem_folder, param, total_Area=None):
    """
    Runs convergence tests over a series of mesh sizes

    Inputs:
    ------
        mesh          : array of mesh suffixes
        problem_folder: str, name of folder containing meshes, etc...
        param         : str, name of param file
        total_Area    : float, total area of the meshes involved. (Provide 
                        when avg_density needed for convergence test)

    Returns:
    -------
        N         : len(mesh) array, elements of problem.
        iterations: len(mesh) array, number of iterations to converge.
        Cext_0    : len(mesh) array of float, Cross extinction section of the 
                    main sphere.
        Time      : len(mesh) array of float, time to solution (wall-time)
    """
    print('Runs lspr case of silver sphere in water medium')
    N = numpy.zeros(len(mesh))
    iterations = numpy.zeros(len(mesh))
    avg_density = ['NA']*4
    Cext_0 = numpy.zeros(len(mesh))
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
            if total_Area:
                avg_density[i] = results['total_elements']/total_Area
            iterations[i] = results['iterations']
            Cext_0[i] = results.get('Cext_0') #We do convergence analysis in the main sphere
            Time[i] = results['total_time']
                 

        except (pycuda._driver.MemoryError, pycuda._driver.LaunchError) as e:
            print('Mesh {} failed due to insufficient memory.'
                  'Skipping this test, but convergence test should still complete'.format(mesh[i]))
            time.sleep(4)

    if total_Area:
        mesh_rate = mesh_ratio(avg_density)
        expected_rate = 0
    else: 
        mesh_rate = mesh_ratio(N)
        expected_rate = 0

    if all(ratio==mesh_rate[0] for ratio in mesh_rate):
        expected_rate = mesh_rate[0]
    else:
        print('Mesh ratio inconsistency. \nCheck that the mesh ratio' 
              'remains constant along refinement'
              'Convergence test report will bad convergence for this reason')


    return(N, avg_density, iterations, expected_rate, Cext_0, Time)

def richardson_extrapolation_lspr(test_result):
    """
    Performs an estimate of the exact solution using
    Richardson extrapolation, given by

    f_ex = (f_1 * f_3 - f_2^2) / (f_3 - 2*f_2+f_1)

    where f_1 is a result from the finest grid and f_3 is from the coarsest.
    The grids f_1, f_2, f_3 should have the same refinement ratio (e.g. 2 -> 4 -> 8)

    Arguments:
    ----------
    test_result:

    Returns:
    --------
    f_ex : float, richardson_extrapolation estimated exact solution.  
    """
    #We perform the richardson extrapolation in the main body. The body we
    #meassure
    try:
        Cext_0 = test_result['Cext_0']
    except KeyError:
        print('No results found for main body cross extinction section  \n'
              'Something has gone wrong.')
        sys.exit()

    # assuming 4 runs
    f1 = Cext_0[3] 
    f2 = Cext_0[2]
    f3 = Cext_0[1]

    f_ex = (f1 * f3 - f2**2) / (f3 - 2 * f2 + f1)

    return f_ex 

