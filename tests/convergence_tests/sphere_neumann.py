from pygbe.util import an_solution
from convergence import (run_convergence, picklesave, pickleload,
                         report_results, mesh)


def main():
    print('{:-^60}'.format('Running sphere_neumann test'))
    try:
        test_outputs = pickleload()
    except FileNotFoundError:
        test_outputs = {}

    problem_folder = 'input_files'

    # neumann_surface
    param = 'sphere_fine.param'
    test_name = 'neumann_surface'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # load results for analysis
    Esolv, Esurf, Ecoul = test_outputs['neumann_surface'][2:5]
    Time = test_outputs['neumann_surface'][-1]
    N, iterations = test_outputs['neumann_surface'][:2]

    Etotal = Esolv + Esurf + Ecoul
    total_time = Time

    analytical = an_solution.constant_charge_single_energy(-80 * 1, 4, 0.125,
                                                           80)

    error = abs(Etotal - analytical) / abs(analytical)

    report_results(error,
                   N,
                   iterations,
                   Etotal,
                   analytical,
                   total_time,
                   energy_type='Total',
                   test_name='sphere neumann')


if __name__ == "__main__":
    from check_for_meshes import check_mesh
    mesh_file = 'https://zenodo.org/record/55349/files/pygbe_regresion_test_meshes.zip'
    folder_name = 'regresion_tests_meshes'
    rename_folder = 'geometry'
    size = '~10MB'
    check_mesh(mesh_file, folder_name, rename_folder, size)
    main()