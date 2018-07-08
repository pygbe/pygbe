"""
This is the main function of the program.
We use a boundary element method (BEM) to perform simple nanoparticle
plasmonics. Localized surface plasmon resonance (LSPR) is an optical effect,
but electrostatics is a good approximation in the long-wavelength limit. For
nanoparticles smaller than the wavelength of incident light, PyGBe 
compute the extinction cross-section.
"""

import os
import sys
import time
import numpy
import pickle
from datetime import datetime

# Import self made modules
import pygbe
from pygbe.gmres import gmres_mgs
from pygbe.classes import Timing, Parameters, IndexConstant
from pygbe.gpuio import dataTransfer
from pygbe.class_initialization import initialize_surface, initialize_field
from pygbe.output import print_summary
from pygbe.matrixfree import (generateRHS, generateRHS_gpu, dipole_moment,
                              extinction_cross_section)
from pygbe.util.read_data import read_parameters, read_electric_field
from pygbe.tree.FMMutils import computeIndices, precomputeTerms, generateList
from pygbe.main import (Logger, read_inputs, find_config_files, check_for_nvcc)    

try:
    from pygbe.tree.cuda_kernels import kernels
except:
    pass


def main(argv=sys.argv, log_output=True, return_output_fname=False,
         return_results_dict=False, field=None, lspr_values=None):
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
         programmatically stepping through slight changes in a problem.
    lspr_values : list, defaults to None
         If passed, provides values for `electric_field` and `wavelength`, useful to
         programmatically step through varying wavelengths

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

    ### Read electric field and its wavelength.
    if lspr_values:
        electric_field, wavelength = lspr_values
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
    if complex_diel:
        phi = numpy.zeros(param.Neq, dtype=numpy.complex)
    else:
        raise ValueError('Dielectric should be complex for LSPR problems')

    phi, iteration = gmres_mgs(surf_array, field_array, phi, F, param, ind0,
                            timing, kernel)
    toc = time.time()

    results_dict['iterations'] = iteration
    solve_time = toc - tic
    print('Solve time        : {}s'.format(solve_time))
    phifname = '{:%Y-%m-%d-%H%M%S}-phi_{}.txt'.format(datetime.now(), wavelength)
    results_dict['solve_time'] = solve_time
    numpy.savetxt(os.path.join(output_dir, phifname), phi)

    # Put result phi in corresponding surfaces
    s_start = 0
    for surf in surf_array:
        s_start = surf.fill_phi(phi, s_start)


    #Calculate extinction cross section for lspr problems
    if abs(electric_field) > 1e-12:

        ###Calculating the dipole moment
        dipole_moment(surf_array, electric_field)
 
        print('\nCalculate extinction cross section (Cext)')
        tic = time.time()
        Cext, surf_Cext = extinction_cross_section(surf_array, numpy.array([1,0,0]), numpy.array([0,0,1]),
                           wavelength, electric_field)
        toc = time.time()
        print('Time Cext: {}s'.format(toc - tic))

        print('\nWavelength: {:.2f} nm'.format(wavelength/10))
        print('Incoming Electric Field: {:.4f} e/(Ang^2 eps_0)'.format(electric_field))

        print('\nCext per surface')
        for i in range(len(Cext)):
            print('Surface {}: {} nm^2'.format(surf_Cext[i], Cext[i]))

        results_dict['time_Cext'] = toc - tic
        results_dict['surf_Cext'] = surf_Cext
        results_dict['wavelength'] = wavelength
        results_dict['E_field'] = electric_field
        results_dict['Cext_list'] = Cext
        results_dict['Cext_0'] = Cext[0]   #We do convergence analysis in the main sphere

    else:
        raise ValueError('electric_field should not be zero to calculate'
                         'the extinction cross section')

    results_dict['total_time'] = (toc - TIC)
    results_dict['version'] = pygbe.__version__

    output_pickle = outputfname.split('-')
    output_pickle.pop(-1)
    output_pickle.append('resultspickle_'+str(wavelength))
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
