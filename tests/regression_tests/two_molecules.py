import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pickle

from pygbe.util import an_solution
from regression import (scanOutput, run_regression, picklesave, pickleload,
                        report_results, mesh)


def main():
    print('{:-^60}'.format('Running two_molecules test'))
    try:
        test_outputs = pickleload()
    except IOError:
        test_outputs = {}

    problem_folder = 'input_files'

    #twosphere
    print('Runs for two molecules')
    param = 'sphere_fine.param'
    test_name = 'twosphere'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name,
            problem_folder,
            param, delete_output=False)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    #molecule_single_center
    print('Runs for isolated molecule')
    param = 'sphere_fine.param'
    test_name = 'molecule_single_center'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name,
            problem_folder,
            param, delete_output=False)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    #load results for analysis
    Esolv, Esurf, Ecoul = test_outputs['twosphere'][2:5]
    Esolv_single, Esurf_single, Ecoul_single = test_outputs[
        'molecule_single_center'][2:5]
    Time = test_outputs['twosphere'][-1]
    Time_single = test_outputs['molecule_single_center'][-1]
    N, iterations = test_outputs['twosphere'][:2]

    Etotal = Esolv + Esurf + Ecoul
    total_time = Time
    Esolv_single *= 2  # Same molecule twice

    Einter = Esolv + Esurf + Ecoul - Esurf_single - Ecoul_single - Esolv_single
    total_time = Time + Time_single

    analytical, EE1, EE2 = an_solution.two_sphere(5., 12., 0.125, 4., 80., 1.)
    analytical *= 2

    error = abs(Einter - analytical) / abs(analytical)

    report_results(error, N, iterations, Einter, analytical, total_time)

if __name__ == "__main__":
    from check_for_meshes import check_mesh
    check_mesh()
    main()
