import re
import os
import shutil
import numpy
import pickle

from pygbe.main import main as pygbe


ITER_REGEX = re.compile('Converged after (\d*) iterations')
N_REGEX = re.compile('Total elements : (\d*)')
ESOLV_REGEX = re.compile('Esolv = (\-*\d*\.\d*)\ kcal\/mol')
ESURF_REGEX = re.compile('[^: ]Esurf = (\-*\d*\.\d*)\ kcal\/mol')
ECOUL_REGEX = re.compile('Ecoul = (\-*\d*\.\d*)\ kcal\/mol')
TIME_REGEX = re.compile('Time = (\-*\d*\.\d*)\ s')

def picklesave(test_outputs):
    with open('tests','w') as f:
        pickle.dump(test_outputs, f)

def pickleload():
    with open('tests', 'r') as f:
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



def run_regression(mesh, test_name, problem_folder, param, delete_output=True):
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
                         '-c', '{}_{}.config'.format(test_name, mesh[i]),
                         '-o', 'output_{}_{}'.format(test_name, mesh[i]),
                         '-g', '../../pygbe/',
                         '{}'.format(problem_folder),], return_output_fname=True)

        print 'Scan output file'
        outfolder = os.path.join('{}'.format(problem_folder),
                                 'output_{}_{}'.format(test_name, mesh[i]))
        outfile = os.path.join(outfolder, outfile)
        N[i],iterations[i],Esolv[i],Esurf[i],Ecoul[i],Time[i] = scanOutput(outfile)
        if delete_output:
            shutil.rmtree(outfolder)


    return(N, iterations, Esolv, Esurf, Ecoul, Time)
