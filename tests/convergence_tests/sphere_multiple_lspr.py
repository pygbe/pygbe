import numpy
from convergence_lspr import (run_convergence, picklesave, pickleload,
                              report_results, richardson_extrapolation_lspr,
                              mesh_multiple)

def main():
    print('{:-^60}'.format('Running sphere_multiple_lspr test'))
    try:
        test_outputs = pickleload()
    except FileNotFoundError:
        test_outputs = {}

    
    problem_folder = 'input_files'

    # multiple spheres lspr 
    param = 'sphere_complex.param'
    test_name = 'sphere_multiple_complex'

    #In this case we have 1 sphere of radius 10 and 6 of radius 2
    R=10.
    r=4.    
    total_Area = 4*numpy.pi*(R*R + 6*r*r)

    if test_name not in test_outputs.keys():
       N, avg_density, iterations, expected_rate, Cext_0, Time = run_convergence(
            mesh_multiple, test_name, problem_folder, param, total_Area=total_Area)
       test_outputs[test_name] = {'N': N,
                                  'avg_density': avg_density, 
                                  'iterations': iterations,
                                  'expected_rate': expected_rate,
                                  'Cext_0': Cext_0,
                                  'Time': Time} 


    # load data for analysis
    N = test_outputs[test_name]['N']
    avg_density = test_outputs[test_name]['avg_density']
    iterations = test_outputs[test_name]['iterations']
    expected_rate = test_outputs[test_name]['expected_rate']
    Cext_0 = test_outputs[test_name]['Cext_0']
    Time = test_outputs[test_name]['Time']

    total_time = Time

    #Richardson extrapolation on the main sphere:
    rich_extra = richardson_extrapolation_lspr(test_outputs[test_name])  

    error = abs(Cext_0 - rich_extra) / abs(rich_extra)

    test_outputs[test_name]['error'] = error
    test_outputs[test_name]['rich_extra'] = rich_extra

    picklesave(test_outputs)
 
    report_results(error,
                       N,
                       expected_rate,
                       iterations,
                       Cext_0,
                       total_time,
                       analytical=None,
                       rich_extra=rich_extra,
                       avg_density=avg_density,
                       test_name=test_name)

if __name__ == "__main__":
    from check_for_meshes import check_mesh
    mesh_file = ''
    folder_name = 'lspr_convergence_test_meshes'
    rename_folder = 'geometry_lspr'
    size = '~3MB'
    check_mesh(mesh_file, folder_name, rename_folder, size)
    main()
