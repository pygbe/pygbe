from pygbe.util import an_solution
from convergence import (run_convergence, picklesave, pickleload,
                         report_results, mesh)


def main():
    print('{:-^60}'.format('Running molecule_neumann test'))

    try:
        test_outputs = pickleload()
    except FileNotFoundError:
        test_outputs = {}

    problem_folder = 'input_files'

    # molecule_neumann
    print('Runs for molecule + set phi/dphi surface')
    param = 'sphere_fine.param'
    test_name = 'molecule_neumann'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # molecule_single_center
    print('Runs for isolated molecule')
    param = 'sphere_fine.param'
    test_name = 'molecule_single_center'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # neumann_surface
    print('Runs for isolated surface')
    param = 'sphere_fine.param'
    test_name = 'neumann_surface'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # Load results for analysis
    Esolv, Esurf, Ecoul = test_outputs['molecule_neumann'][2:5]
    Esolv_mol, Esurf_mol, Ecoul_mol = test_outputs['molecule_single_center'][2:
                                                                             5]
    Esolv_surf, Esurf_surf, Ecoul_surf = test_outputs['neumann_surface'][2:5]
    Time = test_outputs['molecule_neumann'][-1]
    Time_mol = test_outputs['molecule_single_center'][-1]
    Time_surf = test_outputs['neumann_surface'][-1]
    N, iterations = test_outputs['molecule_neumann'][:2]

    Einter = (Esolv + Esurf + Ecoul - Esolv_surf - Esurf_mol - Ecoul_mol -
              Esolv_mol - Esurf_surf - Ecoul_surf)
    total_time = Time + Time_mol + Time_surf

    analytical = an_solution.molecule_constant_charge(1., -80 * 1., 5., 4.,
                                                      12., 0.125, 4., 80.)

    error = abs(Einter - analytical) / abs(analytical)

    report_results(error, N, iterations, Einter, analytical, total_time,
                   test_name='molecule neumann')


if __name__ == "__main__":
    from check_for_meshes import check_mesh
    mesh_file = 'https://zenodo.org/record/55349/files/pygbe_regresion_test_meshes.zip'
    folder_name = 'regresion_tests_meshes'
    rename_folder = 'geometry'
    size = '~10MB'
    check_mesh(mesh_file, folder_name, rename_folder, size)
    main()