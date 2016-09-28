import re
import os
import sys
import time
import numpy
import shutil
import pickle

try:
    import pycuda
except ImportError:
    ans = input('PyCUDA not found.  Regression tests will take forever.  Do you want to continue? [y/n] ')
    if ans in ['Y', 'y']:
        pass
    else:
        sys.exit()

from pygbe.main import main as pygbe


ITER_REGEX = re.compile('Converged after (\d*) iterations')
N_REGEX = re.compile('Total elements : (\d*)')
ESOLV_REGEX = re.compile('[^: ]Esolv = (\-*\d*\.\d*)\ kcal\/mol')
ESURF_REGEX = re.compile('[^: ]Esurf = (\-*\d*\.\d*)\ kcal\/mol')
ECOUL_REGEX = re.compile('[^: ]Ecoul = (\-*\d*\.\d*)\ kcal\/mol')
TIME_REGEX = re.compile('Time = (\-*\d*\.\d*)\ s')

mesh = ['500', '2K', '8K', '32K']
lysozome_mesh = ['1','2','4','8']

def picklesave(test_outputs):
    with open('tests','wb') as f:
        pickle.dump(test_outputs, f, 2)

def pickleload():
    with open('tests', 'rb') as f:
        test_outputs = pickle.load(f)

    return test_outputs

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


def report_results(error, N, iterations, E, analytical, total_time, energy_type='Interaction'):
    """
    Prints out information for the convergence tests.

    Inputs:
    -------
        error: list of float
            L2 Norm of error against analytical solution
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

    flag = 0
    for i in range(len(error)-1):
        rate = error[i]/error[i+1]
        if abs(rate-4)>0.6:
            flag = 1
            print('Bad convergence for mesh {} to {}, with rate {}'.
                  format(i, i+1, rate))

    if flag==0:
        print('Passed convergence test!')

    print('\nNumber of elements : {}'.format(N))
    print('Number of iteration: {}'.format(iterations))
    print('{} energy : {}'.format(energy_type, E))
    print('Analytical solution: {} kcal/mol'.format(analytical))
    print('Error              : {}'.format(error))
    print('Total time         : {}'.format(total_time))



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
            outfile = pygbe(['',
                            '-p', '{}'.format(param),
                            '-c', '{}_{}.config'.format(test_name, mesh[i]),
                            '-o', 'output_{}_{}'.format(test_name, mesh[i]),
                            '-g', './',
                            '{}'.format(problem_folder),], return_output_fname=True)

            print('Scan output file')
            outfolder = os.path.join('{}'.format(problem_folder),
                                    'output_{}_{}'.format(test_name, mesh[i]))
            outfile = os.path.join(outfolder, outfile)
            N[i],iterations[i],Esolv[i],Esurf[i],Ecoul[i],Time[i] = scanOutput(outfile)
            if delete_output:
                shutil.rmtree(outfolder)

        except (pycuda._driver.MemoryError, pycuda._driver.LaunchError) as e:
            print('Mesh {} failed due to insufficient memory.'
                  'Skipping this test, but convergence test should still complete'.format(mesh[i]))
            time.sleep(4)


    return(N, iterations, Esolv, Esurf, Ecoul, Time)
