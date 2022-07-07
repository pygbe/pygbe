"""
This is the main function of the program.
We use a boundary element method (BEM) to perform molecular electrostatics
calculations with a continuum approach. It calculates solvation energies for
proteins modeled with any number of dielectric regions.
"""
import os
import re
import sys
import time
import glob
import numpy
import pickle
import subprocess
from datetime import datetime
from argparse import ArgumentParser

# Import self made modules
import pygbe
from pygbe.gmres import gmres_mgs
from pygbe.classes import Timing, Parameters, IndexConstant
from pygbe.gpuio import dataTransfer
from pygbe.class_initialization import initialize_surface, initialize_field
from pygbe.output import print_summary
from pygbe.matrixfree import (generateRHS, generateRHS_gpu, calculate_solvation_energy,
                              coulomb_energy, calculate_surface_energy)
from pygbe.util.read_data import read_parameters, read_electric_field
from pygbe.tree.FMMutils import computeIndices, precomputeTerms, generateList

try:
    from pygbe.tree.cuda_kernels import kernels
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

    def flush(self):
        """Required for Python 3"""
        pass


def read_inputs(args):
    """
    Parse command-line arguments to determine which config and param files to run
    Assumes that in the absence of specific command line arguments that pygbe
    problem folder resembles the following structure

    lys/
    - lys.param
    - lys.config
    - built_parse.pqr
    - geometry/Lys1.face
    - geometry/Lys1.vert
    - output/
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
    parser.add_argument('-x0',
                        '--initial_guess',
                        dest='initial_guess',
                        type=str,
                        help="File containing an initial guess for the linear solver")

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

    if cliargs.config is None:
        cliargs.config = next(glob.iglob(os.path.join(full_path, '*.config')))
    else:
        cliargs.config = resolve_relative_config_file(cliargs.config, full_path)
    if cliargs.param is None:
        cliargs.param = next(glob.iglob(os.path.join(full_path, '*.param')))
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
        return True
    except subprocess.CalledProcessError:
        print(
            "Could not find `nvcc` on your PATH.  Is cuda installed?  "
            "PyGBe will continue to run but will run significantly slower.  "
            "For optimal performance, add `nvcc` to your PATH"
        )
        return False


def main(argv=sys.argv, log_output=True, return_output_fname=False,
         return_results_dict=False, field=None, electric_values=None):
    """
    Run a PyGBe problem, write outputs to STDOUT and to log file in
    problem directory

    Arguments
    ----------
    log_output : Bool, default True.
        If False, output is written only to STDOUT and not to a log file.
    return_output_fname: Bool, default False.
        If True, function main() returns the name of the
        output log file. This is used for the regression tests.
    return_results_dict: Bool, default False.
        If True, function main() returns the results of the run
        packed in a dictionary.  Used in testing and programmatic
        use of PyGBe
    field : Dictionary, defaults to None.
         If passed, this dictionary will supercede any config file found, useful in
         programmatically stepping through slight changes in a problem
    electric_values : list, defaults to None
         If passed, provides values for `electric_field`.

    Returns
    --------
    output_fname       : str, if kwarg is True.
                         The name of the log file containing problem output
    """

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
    else:
        geo_path = os.path.join(full_path, 'geometry')

    #try to expand ~ if present in output path
    args.output = os.path.expanduser(args.output)
    #if output path is absolute, use that, otherwise prepend
    #problem path
    if not os.path.isdir(args.output):
        output_dir = os.path.join(full_path, args.output)
    else:
        output_dir = args.output
    # create output directory if it doesn't already exist
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    results_dict = {}
    timestamp = time.localtime()
    outputfname = '{:%Y-%m-%d-%H%M%S}-output.log'.format(datetime.now())
    results_dict['output_file'] = outputfname
    if log_output:
        restore_stdout = sys.stdout
        sys.stdout = Logger(os.path.join(output_dir, outputfname))
    # Time stamp
    print('Run started on:')
    print('\tDate: {}/{}/{}'.format(timestamp.tm_year, timestamp.tm_mon,
                                    timestamp.tm_mday))
    print('\tTime: {}:{}:{}'.format(timestamp.tm_hour, timestamp.tm_min,
                                    timestamp.tm_sec))
    print('\tPyGBe version: {}'.format(pygbe.__version__))
    TIC = time.time()

    print('Config file: {}'.format(configFile))
    print('Parameter file: {}'.format(paramfile))
    print('Geometry folder: {}'.format(geo_path))
    print('Running in: {}'.format(full_path))
    results_dict['config_file'] = configFile
    results_dict['param_file'] = paramfile
    results_dict['geo_file'] = geo_path
    results_dict['full_path'] = full_path

    ### Read parameters
    param = Parameters()
    precision = read_parameters(param, paramfile)

    param.Nm = (param.P + 1) * (param.P + 2) * (
        param.P + 3) // 6  # Number of terms in Taylor expansion
    param.BlocksPerTwig = int(numpy.ceil(param.NCRIT / float(param.BSZ))
                              )  # CUDA blocks that fit per twig

    HAS_GPU = check_for_nvcc()
    if param.GPU == 1 and not HAS_GPU:
        print('\n\n\n\n')
        print('{:-^{}}'.format('No GPU DETECTED', 60))
        print("Your param file has `GPU = 1` but CUDA was not detected.\n"
              "Continuing using CPU.  If you do not want this, use Ctrl-C\n"
              "to stop the program and check that your CUDA installation\n"
              "is on your $PATH")
        print('{:-^{}}'.format('No GPU DETECTED', 60))
        print('\n\n\n\n')
        param.GPU = 0

    ### Generate array of fields
    if field:
        field_array = initialize_field(configFile, param, field)
    else:
        field_array = initialize_field(configFile, param)


    ### Generate array of surfaces and read in elements
    surf_array = initialize_surface(field_array, configFile, param)

    ### Read external electric field (potential if kappa>1e-12).  
    if electric_values:
        electric_field, wavelength = electric_values
    else:
        electric_field, wavelength = read_electric_field(param, configFile)


    ### Fill surface class
    time_sort = 0.
    for i in range(len(surf_array)):
        time_sort += surf_array[i].fill_surface(param)

    ### Output setup summary
    param.N = 0
    param.Neq = 0
    for s in surf_array:
        N_aux = len(s.triangle)
        param.N += N_aux        
        if s.surf_type in ['dirichlet_surface', 'neumann_surface', 'asc_surface']:
            param.Neq += N_aux
        else:
            param.Neq += 2 * N_aux
    print('\nTotal elements : {}'.format(param.N))
    print('Total equations: {}'.format(param.Neq))
    print('External potential: {} e/(Ang eps_0)'.format(electric_field))

    results_dict['total_elements'] = param.N
    results_dict['N_equation'] = param.Neq

    results_dict = print_summary(surf_array, field_array, param, results_dict)

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
    print('Generate interaction list')
    tic = time.time()
    generateList(surf_array, field_array, param)
    toc = time.time()
    list_time = toc - tic

    ### Transfer data to GPU
    print('Transfer data to GPU')
    tic = time.time()
    if param.GPU == 1:
        dataTransfer(surf_array, field_array, ind0, param, kernel)
    toc = time.time()
    transfer_time = toc - tic

    timing = Timing()

    ### Generate RHS
    print('Generate RHS')
    tic = time.time()
    if param.GPU == 0:
        F = generateRHS(field_array, surf_array, param, kernel, timing, ind0, electric_field)
    elif param.GPU == 1:
        F = generateRHS_gpu(field_array, surf_array, param, kernel, timing,
                            ind0, electric_field)
    toc = time.time()
    rhs_time = toc - tic

    setup_time = toc - TIC
    print('List time          : {}s'.format(list_time))
    print('Data transfer time : {}s'.format(transfer_time))
    print('RHS generation time: {}s'.format(rhs_time))
    print('-'*30)
    print('Total setup time   : {}s'.format(setup_time))


    #   Check if there is a complex dielectric
    if any([numpy.iscomplexobj(f.E) for f in field_array]):
        complex_diel = True
    else: 
        complex_diel = False

    ### Solve
    tic = time.time()

    print('Solve')
    # Initializing phi dtype according to the problem we are solving.
    if not complex_diel:
        if args.initial_guess==None:
            phi = numpy.zeros(param.Neq)    
        else:
            phi = numpy.loadtxt(args.initial_guess)
            if len(phi) != param.Neq:
                print("Initial guess dimensions don't agree. Continuing with 0")
                phi = numpy.zeros(param.Neq)    

            
    else:
        raise ValueError('Dielectric should be real for solvation energy problems')

#   phi = numpy.loadtxt('examples/zika/phi_reference')
        
    phi, iteration = gmres_mgs(surf_array, field_array, phi, F, param, ind0,
                            timing, kernel)
    toc = time.time()

    results_dict['iterations'] = iteration
    solve_time = toc - tic
    print('Solve time        : {}s'.format(solve_time))
    phifname = '{:%Y-%m-%d-%H%M%S}-phi.txt'.format(datetime.now())
    results_dict['solve_time'] = solve_time
    numpy.savetxt(os.path.join(output_dir, phifname), phi)

    # Data inverse Debye length
    for data in field_array:
        if data.LorY == 2:
            LorY = data.LorY
            kappa = data.kappa
    
    # Put result phi in corresponding surfaces.
    s_start = 0
    for surf in surf_array:
        if abs(electric_field) > 1e-12:
            if LorY == 2 and kappa > 1e-12:
                phi_electric_field = electric_field*numpy.exp(-kappa*abs(surf.zi))
                derphi_electric_field = -electric_field*kappa*numpy.exp(-kappa*abs(surf.zi))
                s_start = surf.fill_phi(phi, s_start, phi_electric_field, derphi_electric_field)
            else:
                phi_electric_field = -electric_field*surf.zi
                derphi_electric_field = -electric_field*surf.normal[:,2]
                s_start = surf.fill_phi(phi, s_start, phi_electric_field, derphi_electric_field)
        else:
            s_start = surf.fill_phi(phi, s_start)


    # Calculate solvation energy
    print('\nCalculate Solvation Energy (E_solv)')
    tic = time.time()
    E_solv = calculate_solvation_energy(surf_array, field_array, param, kernel, output_dir)
    toc = time.time()
    print('Time E_solv: {}s'.format(toc - tic))
    ii = -1
    for i, f in enumerate(field_array):
        if f.pot == 1:
            parent_type = surf_array[f.parent[0]].surf_type
            if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface':
                ii += 1
                print('Region {}: E_solv = {} kcal/mol = {} kJ/mol'.format(i,
                                                                      E_solv[ii],
                                                                      E_solv[ii] * 4.184))


    # Calculate surface energy
    print('\nCalculate Surface Energy (E_surf)')

    tic = time.time()
    E_surf = calculate_surface_energy(surf_array, field_array, param, kernel)
    toc = time.time()
    ii = -1
    for f in param.E_field:
        parent_type = surf_array[field_array[f].parent[0]].surf_type
        if parent_type == 'dirichlet_surface' or parent_type == 'neumann_surface':
            ii += 1
            print('Region {}: E_surf = {} kcal/mol = {} kJ/mol'.format(
                f, E_surf[ii], E_surf[ii] * 4.184))
    print('Time E_surf: {}s'.format(toc - tic))

    ### Calculate Coulombic interaction
    print('\nCalculate Coulomb Energy (E_coul)')
    tic = time.time()
    i = -1
    E_coul = []
    for f in field_array:
        i += 1
        if f.coulomb == 1:
            print('Calculate Coulomb energy for region {}'.format(i))
            E_coul.append(coulomb_energy(f, param))
            print('Region {}: E_coul = {} kcal/mol = {} kJ/mol'.format(
                i, E_coul[-1], E_coul[-1] * 4.184))
    toc = time.time()
    print('Time E_coul: {}s'.format(toc - tic))

    ### Output summary
    print('\n'+'-'*30)
    print('Totals:')
    print('E_solv = {} kcal/mol'.format(sum(E_solv)))
    print('E_surf = {} kcal/mol'.format(sum(E_surf)))
    print('E_coul = {} kcal/mol'.format(sum(E_coul)))
    print('\nTime = {} s'.format(toc - TIC))

    results_dict['E_solv_kcal'] = sum(E_solv)
    results_dict['E_solv_kJ'] = sum(E_solv) * 4.184
    results_dict['E_surf_kcal'] = sum(E_surf)
    results_dict['E_surf_kJ'] = sum(E_surf) * 4.184
    results_dict['E_coul_kcal'] = sum(E_coul)
    results_dict['E_coul_kJ'] = sum(E_coul) * 4.184

    results_dict['total_time'] = (toc - TIC)
    results_dict['version'] = pygbe.__version__

    output_pickle = outputfname.split('-')
    output_pickle.pop(-1)
    output_pickle.append('resultspickle')
    output_pickle = '-'.join(output_pickle)
    with open(os.path.join(output_dir, output_pickle), 'wb') as f:
        pickle.dump(results_dict, f, 2)

    try: 
        with open(os.path.join(output_dir, output_pickle), 'rb') as f:
            pickle.load(f)
    except EOFError:
        print('Error writing the pickle file, the results will be unreadable')
        pass     
    
    #reset stdout so regression tests, etc, don't get logged into the output
    #file that they themselves are trying to read
    if log_output:
        sys.stdout = restore_stdout

    if return_results_dict:
        return results_dict

    if return_output_fname and log_output:
        return outputfname


if __name__ == "__main__":
    sys.exit(main(sys.argv))
