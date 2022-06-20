import os
import pickle
import pytest
import functools

import sys
import atexit

def report_e():
    print('PGBmut sensor test could not run using GPU because pycuda were not found. CPU were used instead.')
def report_g():
    print('PGBmut sensor test had run using GPU.')
def report_c():
    print('PGBmut sensor test had run using CPU.')

from pygbe.main import main

@pytest.mark.parametrize('key', ['total_elements',
                                 'E_solv_kJ',
                                 'E_coul_kcal',
                                 'E_coul_kJ',
                                 'E_solv_kcal'])
def test_PGB_mut_sensor(key, arc):
    results = get_results(arc)

    with open('pgbmut_extfield.pickle', 'rb') as f:
        base_results = pickle.load(f)
    if arc == 'gpu':
        assert abs(base_results[key] - results[key]) / abs(base_results[key]) < 1e-6
    elif arc == 'cpu':
        assert abs(base_results[key] - results[key]) / abs(base_results[key]) < 1e-6

def test_pgbmut_iterations(arc):
    results = get_results(arc)
    with open('pgbmut_extfield.pickle', 'rb') as f:
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

    print('Generating results for 1PGBmut example...')
    if os.getcwd().rsplit('/', 1)[1] == 'tests':
        results = main(['','../examples/1PGBmut_sensor_Extfield'],
                        log_output=False,
                        return_results_dict=True)
    elif os.getcwd().rsplit('/', 1)[1] == 'pygbe':
        results = main(['','./examples/1PGBmut_sensor_Extfield'],
                        log_output=False,
                        return_results_dict=True)
    else:
        print("Run tests from either the main repo directory or the tests directory")

    return results
