import os
import pickle
import pytest
import functools

import sys
import atexit

def report_e():
    print('LSPR 7 Spheres test could not run using GPU because pycuda were not found. CPU were used instead.')
def report_g():
    print('LSPR 7 Spheres test had run using GPU.')
def report_c():
    print('LSPR 7 Spheres test had run using CPU.')

from pygbe.lspr import main


@pytest.mark.parametrize('key_int', ['total_elements',
                                 'iterations'])
def test_lspr_elements_iterations(key_int, arc):
    results = get_results(arc)
    with open('lspr_7_spheres.pickle', 'rb') as f:
        base_results = pickle.load(f)

    assert base_results[key_int] == results[key_int]


@pytest.mark.parametrize('key', ['Cext_0'])
def test_lspr(key, arc):
    results = get_results(arc)

    with open('lspr_7_spheres.pickle', 'rb') as f:
        base_results = pickle.load(f)
    #Cext and surf_Cext are lists, for the example are one element lists, so
    #to check the assertion we access that element. i.e [0]
    assert abs(base_results[key] - results[key]) / abs(base_results[key] + 1e-16) < 1e-9

@functools.lru_cache(4)
def get_results(arc):

    if arc == 'gpu':
        try:
            import pycuda
            if sys.stdout != sys.__stdout__:
                 atexit.register(report_g)
        except ImportError:
            if sys.stdout != sys.__stdout__:
                atexit.register(report_e)
    elif arc == 'cpu':
        if sys.stdout != sys.__stdout__:
            atexit.register(report_c)

    print('Generating results for lspr_7_spheres example...')
    if os.getcwd().rsplit('/', 1)[1] == 'tests':
        results = main(['','../examples/lspr_7_spheres'],
                        log_output=False,
                        return_results_dict=True)
    elif os.getcwd().rsplit('/', 1)[1] == 'pygbe':
        results = main(['','./examples/lspr_7_spheres'],
                        log_output=False,
                        return_results_dict=True)
    else:
        print("Run tests from either the main repo directory or the tests directory")

    return results
