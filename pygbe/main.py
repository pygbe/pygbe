"""
This is the main function of the program.
We use a boundary element method (BEM) to perform molecular electrostatics
calculations with a continuum approach. It calculates solvation energies for
proteins modeled with any number of dielectric regions.
"""
import numpy
import time
from datetime import datetime
import os
import sys
import re
import subprocess
from argparse import ArgumentParser

# Import self made modules
from gmres import gmres_mgs
from projection import get_phir
from classes import Timing, Parameters, IndexConstant
from gpuio import dataTransfer
from surface import initializeSurf, fill_surface, initializeField, fill_phi
from output import printSummary
from matrixfree import (generateRHS, generateRHS_gpu, calculateEsolv,
                        coulombEnergy, calculateEsurf)

from util.readData import (readVertex, readTriangle, readpqr, readParameters,
                          readElectricField)
from util.an_solution import an_P, two_sphere

from tree.FMMutils import computeIndices, precomputeTerms, generateList

try:
    from tree.cuda_kernels import kernels
except:
    pass


#courtesy of http://stackoverflow.com/a/5916874
class Logger(object):
    """
    Allow writing both to STDOUT on screen and sending text to file
    in conjunction with the command
    `sys.stdout = Logger("desired_log_file.txt")`
    """

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


def read_inputs(args):
    """
    Parse command-line arguments to determine which config and param files to run
    Assumes that in the absence of specific command line arguments that pygbe
    problem folder resembles the following structure

    lys
    |- lys.param
    |- lys.config
    |- built_parse.pqr
    |- geometry/Lys1.face
    |- geometry/Lys1.vert
    |- output/
    """

    parser = ArgumentParser(description='Manage PyGBe command line arguments')
    parser.add_argument('problem_folder',
                        type=str,
                        help="Path to folder containing problem files")
    parser.add_argument('-c',
                        '--config',
                        dest='config',
                        type=str,
                        default=None,
                        help="Path to problem config file")
    parser.add_argument('-p',
                        '--param',
                        dest='param',
                        type=str,
                        default=None,
                        help="Path to problem param file")
    parser.add_argument('-o',
                        '--output',
                        dest='output',
                        type=str,
                        default='output',
                        help="Output folder")
    parser.add_argument('-g',
                        '--geometry',
                        dest='geometry',
                        type=str,
                        help="Custom geometry folder prefix")

    return parser.parse_args(args)


def check_file_exists(filename):
    """Try to open the file `filename` and return True if it's valid """
    return os.path.exists(filename)


def find_config_files(cliargs):
    """
    Check that .config and .param files exist and can be opened.
    If either file isn't found, PyGBe exits (and should print which
    file was not found).  Otherwise return the path to the config and
    param files

    Arguments
    ---------
    cliargs: parser
        parser containing cli arguments passed to PyGBe

    Returns
    -------
    cliargs.config: string
        path to config file
    cliargs.param: string
        path to param file
    """

    prob_path = cliargs.problem_folder
    full_path = os.path.abspath(prob_path)
    os.environ['PYGBE_PROBLEM_FOLDER'] = full_path

    #use the name of the rightmost folder in path as problem name
    prob_rel_path = os.path.split(prob_path)
    prob_name = prob_rel_path[1]
    if not prob_name:
        prob_name = os.path.split(prob_rel_path[0])[1]

    if cliargs.config is None:
        cliargs.config = os.path.join(full_path, prob_name + '.config')
    else:
        cliargs.config = resolve_relative_config_file(cliargs.config, full_path)
    if cliargs.param is None:
        cliargs.param = os.path.join(full_path, prob_name + '.param')
    else:
        cliargs.param = resolve_relative_config_file(cliargs.param, full_path)

    return cliargs.config, cliargs.param


def resolve_relative_config_file(config_file, full_path):
    """
    Does its level-headed best to find the config files specified by the user

    Arguments
    ---------
    config_file: str
        the given path to a .param or .config file from the command line
    full_path: str
        the full path to the problem folder
    """

    if check_file_exists(config_file):
        return config_file
    elif check_file_exists(os.path.abspath(config_file)):
        return os.path.join(os.getcwd(), config_file)
    elif check_file_exists(os.path.join(full_path, config_file)):
        return os.path.join(full_path, config_file)
    else:
        sys.exit('Did not find expected config files\n'
                    'Could not find {}'.format(config_file))


def check_for_nvcc():
    """Check system PATH for nvcc, exit if not found"""
    try:
        subprocess.check_output(['which', 'nvcc'])
        check_nvcc_version()
    except subprocess.CalledProcessError:
        print(
            "Could not find `nvcc` on your PATH.  Is cuda installed?  PyGBe will continue to run but will run significantly slower.  For optimal performance, add `nvcc` to your PATH"
        )


def check_nvcc_version():
    """Check that version of nvcc <= 7.0"""
    verstr = subprocess.check_output(['nvcc', '--version'])
    cuda_ver = re.compile('release (\d\.\d)')
    match = re.search(cuda_ver, verstr)
    version = float(match.group(1))
    if version > 7.0:
        sys.exit('PyGBe only supports CUDA <= 7.0\n'
                 'Please install an earlier version of the CUDA toolkit\n'
                 'or remove `nvcc` from your PATH to use CPU only.')


def main(argv=sys.argv, log_output=True, return_output_fname=False):
    """
    Run a PyGBe problem, write outputs to STDOUT and to log file in
    problem directory

    Arguments
    ----------
    log_output         : Bool, default True.
                         If False, output is written only to STDOUT and not
                         to a log file.
    return_output_fname: Bool, default False.
                         If True, function main() returns the name of the
                         output log file. This is used for the regression tests.

    Returns
    --------
    output_fname       : str, if kwarg is True.
                         The name of the log file containing problem output
    """

    check_for_nvcc()

    args = read_inputs(argv[1:])
    configFile, paramfile = find_config_files(args)
    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER')
    #check if a custom geometry location has been specified
    #if it has, add an ENV_VAR to handle it
    if args.geometry:
        geo_path = os.path.abspath(args.geometry)
        if os.path.isdir(geo_path):
            os.environ['PYGBE_GEOMETRY'] = geo_path
        else:
            sys.exit('Invalid geometry prefix provided (Folder not found)')

    #try to expand ~ if present in output path
    args.output = os.path.expanduser(args.output)
    #if output path is absolute, use that, otherwise prepend
    #problem path
    if not os.path.isdir(args.output):
        output_dir = os.path.join(full_path, args.output)
    else:
        output_dir = args.output
    #create output directory if it doesn't already exist
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    timestamp = time.localtime()
    outputfname = '{:%Y-%m-%d-%H%M%S}-output.log'.format(datetime.now())
    if log_output:
        sys.stdout = Logger(os.path.join(output_dir, outputfname))
    ### Time stamp
    print 'Run started on:'
    print '\tDate: %i/%i/%i' % (timestamp.tm_year, timestamp.tm_mon,
                                timestamp.tm_mday)
    print '\tTime: %i:%i:%i' % (timestamp.tm_hour, timestamp.tm_min,
                                timestamp.tm_sec)
    TIC = time.time()

    ### Read parameters
    param = Parameters()
    precision = readParameters(param, paramfile)

    param.Nm = (param.P + 1) * (param.P + 2) * (
        param.P + 3) / 6  # Number of terms in Taylor expansion
    param.BlocksPerTwig = int(numpy.ceil(param.NCRIT / float(param.BSZ))
                              )  # CUDA blocks that fit per twig

    ### Generate array of fields
    field_array = initializeField(configFile, param)

    ### Generate array of surfaces and read in elements
    surf_array = initializeSurf(field_array, configFile, param)

    ### Read electric field and its wavelength.
    electricField, wavelength = readElectricField(param, configFile) 

    ### Fill surface class
    time_sort = 0.
    for i in range(len(surf_array)):
        time_sort += fill_surface(surf_array[i], param)
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    #ss=surf_array[0]
    for i in range(1):
        ss = surf_array[i]
        ax.scatter(ss.xi,ss.yi,ss.zi,c='b',marker='o')
        ax.scatter(ss.xi+ss.normal[:,0], ss.yi+ss.normal[:,1], ss.zi+ss.normal[:,2],c='r', marker='o')
    plt.show()
    quit()
    '''

    ### Output setup summary
    param.N = 0
    param.Neq = 0
    for s in surf_array:
        N_aux = len(s.triangle)
        param.N += N_aux
        if s.surf_type == 'dirichlet_surface' or s.surf_type == 'neumann_surface' or s.surf_type == 'asc_surface':
            param.Neq += N_aux
        else:
            param.Neq += 2 * N_aux
    print '\nTotal elements : %i' % param.N
    print 'Total equations: %i' % param.Neq

    printSummary(surf_array, field_array, param)

    ### Precomputation
    ind0 = IndexConstant()
    computeIndices(param.P, ind0)
    precomputeTerms(param.P, ind0)

    ### Load CUDA code
    if param.GPU == 1:
        kernel = kernels(param.BSZ, param.Nm, param.K_fine, param.P, precision)
    else:
        kernel = 1

    ### Generate interaction list
    print 'Generate interaction list'
    tic = time.time()
    generateList(surf_array, field_array, param)
    toc = time.time()
    list_time = toc - tic

    ### Transfer data to GPU
    print 'Transfer data to GPU'
    tic = time.time()
    if param.GPU == 1:
        dataTransfer(surf_array, field_array, ind0, param, kernel)
    toc = time.time()
    transfer_time = toc - tic

    timing = Timing()

    ### Generate RHS
    print 'Generate RHS'
    tic = time.time()
    if param.GPU == 0:
        F = generateRHS(field_array, surf_array, param, kernel, timing, ind0)
    elif param.GPU == 1:
        F = generateRHS_gpu(field_array, surf_array, param, kernel, timing,
                            ind0)
    toc = time.time()
    rhs_time = toc - tic

    #    numpy.savetxt(os.path.join(output_dir,'RHS.txt'),F)

    setup_time = toc - TIC
    print 'List time          : %fs' % list_time
    print 'Data transfer time : %fs' % transfer_time
    print 'RHS generation time: %fs' % rhs_time
    print '------------------------------'
    print 'Total setup time   : %fs\n' % setup_time


    #   Check if there is a complex dielectric
    complexDiel = 0
    for f in field_array:
        if type(f.E) == complex:
            complexDiel = 1


    ### Solve
    tic = time.time()

    print 'Solve'

    # Initializing phi dtype according to the problem we are solving. 
    if complexDiel == 1:
        phi = numpy.zeros(param.Neq, complex)
    else:    
        phi = numpy.zeros(param.Neq)

    phi = gmres_mgs(surf_array, field_array, phi, F, param, ind0, timing,
                       kernel)
    toc = time.time()
    solve_time = toc - tic

    print 'Solve time        : %fs' % solve_time
    phifname = '{:%Y-%m-%d-%H%M%S}-phi.txt'.format(datetime.now())
    numpy.savetxt(os.path.join(output_dir, phifname), phi)
    #phi = loadtxt('phi.txt')

    # Put result phi in corresponding surfaces
    fill_phi(phi, surf_array)

    ### Calculate solvation energy
    print '\nCalculate Esolv'
    tic = time.time()
    E_solv = calculateEsolv(surf_array, field_array, param, kernel)
    toc = time.time()
    print 'Time Esolv: %fs' % (toc - tic)
    ii = -1
    for f in param.E_field:
        parent_type = surf_array[field_array[f].parent[0]].surf_type
        if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface':
            ii += 1
            print 'Region %i: Esolv = %f kcal/mol = %f kJ/mol' % (
                f, E_solv[ii], E_solv[ii] * 4.184)

    ### Calculate surface energy
    print '\nCalculate Esurf'
    tic = time.time()
    E_surf = calculateEsurf(surf_array, field_array, param, kernel)
    toc = time.time()
    ii = -1
    for f in param.E_field:
        parent_type = surf_array[field_array[f].parent[0]].surf_type
        if parent_type == 'dirichlet_surface' or parent_type == 'neumann_surface':
            ii += 1
            print 'Region %i: Esurf = %f kcal/mol = %f kJ/mol' % (
                f, E_surf[ii], E_surf[ii] * 4.184)
    print 'Time Esurf: %fs' % (toc - tic)

    ### Calculate Coulombic interaction
    print '\nCalculate Ecoul'
    tic = time.time()
    i = -1
    E_coul = []
    for f in field_array:
        i += 1
        if f.coulomb == 1:
            print 'Calculate Coulomb energy for region %i' % i
            E_coul.append(coulombEnergy(f, param))
            print 'Region %i: Ecoul = %f kcal/mol = %f kJ/mol' % (
                i, E_coul[-1], E_coul[-1] * 4.184)
    toc = time.time()
    print 'Time Ecoul: %fs' % (toc - tic)

    ### Output summary
    print '\n--------------------------------'
    print 'Totals:'
    print 'Esolv = %f kcal/mol' % sum(E_solv)
    print 'Esurf = %f kcal/mol' % sum(E_surf)
    print 'Ecoul = %f kcal/mol' % sum(E_coul)
    print '\nTime = %f s' % (toc - TIC)

    #reset stdout so regression tests, etc, don't get logged into the output
    #file that they themselves are trying to read
    sys.stdout = sys.__stdout__

    if return_output_fname and log_output:
        return outputfname


if __name__ == "__main__":
    sys.exit(main(sys.argv))
