import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pickle

from pygbe.util import an_solution
from regression import (scanOutput, run_regression, picklesave, pickleload,
                        report_results, mesh)


def main():
    print('{:-^60}'.format('Running sphere_molecule_single test'))
    try:
        test_outputs = pickleload()
    except IOError:
        test_outputs = {}

    problem_folder = 'input_files'

    #molecule_single
    param = 'sphere_fine.param'
    test_name = 'molecule_single'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    #load results for analysis
    Esolv, Esurf, Ecoul = test_outputs['molecule_single'][2:5]
    Time = test_outputs['molecule_single'][-1]
    N, iterations = test_outputs['molecule_single'][:2]

    total_time = Time

    analytical = an_solution.an_P(
        numpy.array([1.]), numpy.array([[1., 1., 1.41421356]]), 4., 80., 5.,
        0.125, 5., 20)

    error = abs(Esolv - analytical) / abs(analytical)

    report_results(error,
                   N,
                   iterations,
                   Esolv,
                   analytical,
                   total_time,
                   energy_type='Total')


if __name__ == "__main__":
    from check_for_meshes import check_mesh
    check_mesh()
    main()
