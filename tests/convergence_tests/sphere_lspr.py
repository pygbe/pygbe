from pygbe.util import an_solution
from convergence_lspr import (run_convergence, picklesave, pickleload,
                              report_results, mesh)

def main():
    print('{:-^60}'.format('Running sphere_lspr test'))
    try:
        test_outputs = pickleload()
    except FileNotFoundError:
        test_outputs = {}

    #This test is for 10 nm radius silver sphere in water, at wavelength 380 nm
    #and a 10 nm gold sphere in water, at wavelength 380 
    radius = 10.
    wavelength_Ag = 380.
    wavelength_Au = 520.
    diel_out_380 = 1.7972083599999999 + 1j * 8.504766399999999e-09 #water value extrapolated
    diel_out_520 = 1.7800896400000001+ 1j * 3.3515104000000003e-09
    diel_in_Ag = -3.3876520488233184 + 1j * 0.19220746083441781 #silver value extrapolated
    diel_in_Au = -3.8874936460215972+ 1j * 2.6344121588317515 #gold value extrapolated

    analytical_Ag = an_solution.Cext_analytical(radius, wavelength_Ag, diel_out_380, diel_in_Ag)
    analytical_Au = an_solution.Cext_analytical(radius, wavelength_Au, diel_out_520, diel_in_Au)

    problem_folder = 'input_files'

    # single sphere lspr
    param = 'sphere_complex.param'
    test_names = ['sphereAg_complex', 'sphereAu_complex']
    for test_name in test_names:
        if test_name not in test_outputs.keys():
           N, avg_density, iterations, expected_rate, Cext_0, Time = run_convergence(
                mesh, test_name, problem_folder, param, total_Area=None)
           test_outputs[test_name] = {'N': N, 
                                      'iterations': iterations,
                                      'expected_rate': expected_rate,
                                      'Cext_0': Cext_0,
                                      'Time': Time} 
    

        # load data for analysis
        N = test_outputs[test_name]['N']
        iterations = test_outputs[test_name]['iterations']
        expected_rate = test_outputs[test_name]['expected_rate']
        Cext_0 = test_outputs[test_name]['Cext_0']
        Time = test_outputs[test_name]['Time']

        total_time = Time

        if 'Ag' in test_name:
            analytical = analytical_Ag
        elif 'Au' in test_name:
            analytical = analytical_Au

        error = abs(Cext_0 - analytical) / abs(analytical)

        test_outputs[test_name]['error'] = error
        test_outputs[test_name]['analytical'] = analytical

        picklesave(test_outputs)
     
        report_results(error,
                       N,
                       expected_rate,
                       iterations,
                       Cext_0,
                       total_time,
                       analytical=analytical,
                       rich_extra=None,
                       avg_density=None,
                       test_name=test_name)


if __name__ == "__main__":
    from check_for_meshes import check_mesh
    mesh_file = ''
    folder_name = 'lspr_convergence_test_meshes'
    rename_folder = 'geometry_lspr'
    size = '~3MB'
    check_mesh(mesh_file, folder_name, rename_folder, size)
    main()