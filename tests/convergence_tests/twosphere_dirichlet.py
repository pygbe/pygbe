from pygbe.util import an_solution
from convergence import (run_convergence, picklesave, pickleload,
                         report_results, mesh)


def main():
    print('{:-^60}'.format('Running twosphere_dirichlet test'))
    try:
        test_outputs = pickleload()
    except FileNotFoundError:
        test_outputs = {}

    problem_folder = 'input_files'

    # twosphere_dirichlet
    param = 'sphere_fine.param'
    test_name = 'twosphere_dirichlet'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # dirichlet_surface
    param = 'sphere_fine.param'
    test_name = 'dirichlet_surface'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # load results for analysis
    Esolv, Esurf, Ecoul = test_outputs['twosphere_dirichlet'][2:5]
    Esolv_surf, Esurf_surf, Ecoul_surf = test_outputs['dirichlet_surface'][2:5]
    Time = test_outputs['twosphere_dirichlet'][-1]
    Time_surf = test_outputs['dirichlet_surface'][-1]
    N, iterations = test_outputs['twosphere_dirichlet'][:2]

    Einter = Esurf - 2 * Esurf_surf  # Isolated sphere has to be done twice
    total_time = Time + Time_surf

    analytical = an_solution.constant_potential_twosphere_dissimilar(
        1., 1., 4., 4., 12., 0.125, 80.)

    error = abs(Einter - analytical) / abs(analytical)

    report_results(error, N, iterations, Einter, analytical, total_time,
                   test_name='twosphere dirichlet')


if __name__ == "__main__":
    from check_for_meshes import check_mesh
    mesh_file = 'https://zenodo.org/record/55349/files/pygbe_regresion_test_meshes.zip'
    folder_name = 'regresion_tests_meshes'
    rename_folder = 'geometry'
    size = '~10MB'
    check_mesh(mesh_file, folder_name, rename_folder, size)
    main()