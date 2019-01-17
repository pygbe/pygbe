import os
import pickle
import pytest
import functools

import sys
import atexit

def report_e():
    print('Sphere test could not run using GPU because pycuda were not found. CPU were used instead.')
def report_g():
    print('Sphere test had run using GPU.')
def report_c():
    print('Sphere test had run using CPU.')

from pygbe.main import main

@pytest.mark.parametrize('key', ['total_elements',
                                 'E_solv_kJ',
                                 'E_coul_kcal',
                                 'E_coul_kJ',
                                 'E_solv_kcal'])
def test_sphere(key, arc):
    results = get_results(arc)

    with open('sphere.pickle', 'rb') as f:
        base_results = pickle.load(f)

    assert abs(base_results[key] - results[key]) / abs(base_results[key] + 1e-16) < 1e-12

def test_sphere_iterations(arc):
    results = get_results(arc)
    with open('sphere.pickle', 'rb') as f:
        base_results = pickle.load(f)

    assert base_results['iterations'] == results['iterations']

@functools.lru_cache(6)
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

    print('Generating results for sphere example...')
    if os.getcwd().rsplit('/', 1)[1] == 'tests':
        results = main(['','../examples/sphere'],
                        log_output=False,
                        return_results_dict=True)
    elif os.getcwd().rsplit('/', 1)[1] == 'pygbe':
        results = main(['','./examples/sphere'],
                        log_output=False,
                        return_results_dict=True)
    else:
        print("Run tests from either the main repo directory or the tests directory")

    return results
