import numpy
import datetime

from convergence import run_convergence, picklesave, pickleload
from convergence import lysozome_mesh as mesh


def main():
    print('{:-^60}'.format('Running lysozyme test'))
    try:
        test_outputs = pickleload()
    except FileNotFoundError:
        test_outputs = {}

    problem_folder = 'input_files'

    # lys_single
    param = 'lys.param'
    test_name = 'lys_single'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(mesh,
                                                                   test_name,
                                                                   problem_folder,
                                                                   param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # lys
    param = 'lys.param'
    test_name = 'lys'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(mesh,
                                                                   test_name,
                                                                   problem_folder,
                                                                   param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    # lys_k0
    print('Simulations for Lysozyme with kappa=0')
    param = 'lys.param'
    test_name = 'lys_k0'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_convergence(mesh,
                                                                   test_name,
                                                                   problem_folder,
                                                                   param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    Esolv_single, Esurf_single, Ecoul_single = test_outputs['lys_single'][2:5]
    Esolv_full, Esurf_full, Ecoul_full = test_outputs['lys'][2:5]
    Esolv_k0, Esurf_k0, Ecoul_k0 = test_outputs['lys_k0'][2:5]
    iterations_single = test_outputs['lys_single'][1]
    iterations_full = test_outputs['lys'][1]
    iterations_k0 = test_outputs['lys_k0'][1]

    Esolv_ref_single = 1/4.184*numpy.array([-2401.2, -2161.8, -2089, -2065.5])
    Esolv_ref_full = 1/4.184*numpy.array([-2432.9, -2195.9, -2124.2, -2101.1])
    Esolv_FFTSVD = numpy.array([-577.105, -520.53, -504.13, -498.26])  # Remember FFTSVD was only run with kappa=0

    iter_ref_single = numpy.array([33, 34, 35, 39])
    iter_ref_full = numpy.array([36, 38, 41, 45])
    iter_FFTSVD = numpy.array([32, 34, 35, 37])

    error_single = abs(Esolv_single - Esolv_ref_single) / abs(Esolv_ref_single)
    error_full = abs(Esolv_full - Esolv_ref_full) / abs(Esolv_ref_full)
    error_FFTSVD = abs(Esolv_k0 - Esolv_FFTSVD) / abs(Esolv_FFTSVD)

    iter_diff_single = iterations_single - iter_ref_single
    iter_diff_full = iterations_full - iter_ref_full
    iter_diff_FFTSVD = iterations_k0 - iter_FFTSVD

    flag = 0
    thresh = 1e-2
    with open('convergence_test_results', 'a') as f:
        print('-' * 60, file=f)
        print('{:-^60}'.format('Running lysozyme test'), file=f)
        print('-' * 60, file=f)
        print(datetime.datetime.now(), file=f)
        for i in range(len(error_single)):
            if error_single[i] > thresh:
                flag = 1
                print('Solvation energy not agreeing for single surface simulation,'
                      'mesh {} by {}'.format(i, error_single[i]), file=f)

            if error_full[i] > thresh:
                flag = 1
                print('Solvation energy not agreeing for full surface simulation,'
                    'mesh {} by {}'.format(i, error_full[i]), file=f)

            if error_FFTSVD[i] > thresh:
                flag = 1
                print('Solvation energy not agreeing with FFTSVD, mesh {} by'
                    '{}'.format(i, error_FFTSVD[i]), file=f)

        if flag == 0:
            print('\nPassed Esolv test!', file=f)
        else:
            print('\nFAILED Esolv test', file=f)

        flag = 0
        thresh = 3
        for i in range(len(iter_diff_single)):
            if abs(iter_diff_single[i]) > thresh:
                flag = 1
                print('Solvation energy not agreeing for single surface simulation,'
                    'mesh {} by {}'.format(i, iter_diff_single[i]), file=f)

            if abs(iter_diff_full[i]) > thresh:
                flag = 1
                print('Solvation energy not agreeing for full surface simulation, '
                    'mesh {} by {}'.format(i, iter_diff_full[i]), file=f)

            if abs(iter_diff_FFTSVD[i]) > thresh:
                flag = 1
                print('Solvation energy not agreeing with FFTSVD,'
                    'mesh {} by {}'.format(i, iter_diff_FFTSVD[i]), file=f)

        if flag == 0:
            print('\nPassed iterations test! They are all within {} iterations of '
                'reference'.format(thresh), file=f)
        else:
            print('\nFAILED iterations test', file=f)

        print('Summary:', file=f)
        print('Single: Esolv: ', str(Esolv_single),', iterations: ',
              str(iterations_single), file=f)
        print('Full  : Esolv: ', str(Esolv_full), ', iterations: ',
              str(iterations_full), file=f)
        print('k=0   : Esolv: ', str(Esolv_k0), ', iterations: ',
              str(iterations_k0), file=f)

if __name__ == "__main__":
    from check_for_meshes import check_mesh
    mesh_file = 'https://zenodo.org/record/55349/files/pygbe_regresion_test_meshes.zip'
    folder_name = 'regresion_tests_meshes'
    rename_folder = 'geometry'
    size = '~10MB'
    check_mesh(mesh_file, folder_name, rename_folder, size)
    main()
