import os
import pickle
import pytest
import functools

try:
    import pycuda
except ImportError:
    ans = input('PyCUDA not found.  Regression tests will take forever.  Do you want to continue? [y/n] ')
    if ans in ['Y', 'y']:
        pass
    else:
        sys.exit()

from pygbe.main import main


@pytest.mark.parametrize('key_int', ['total_elements',
                                 'iterations'])
def test_lspr_elements_iterations(key_int):
    results = get_results()
    with open('lspr.pickle', 'rb') as f:
        base_results = pickle.load(f)

    assert base_results[key_int] == results[key_int]


@pytest.mark.parametrize('key', ['Cext',
                                 'surf_Cext'])
def test_lspr(key):
    results = get_results()

    with open('lspr.pickle', 'rb') as f:
        base_results = pickle.load(f)
    #Cext and surf_Cext are lists, for the example are one element lists, so
    #to check the assertion we access that element. i.e [0]
    assert abs(base_results[key][0] - results[key][0]) / abs(base_results[key][0] + 1e-16) < 1e-12

@functools.lru_cache(4)
def get_results():
    print('Generating results for lspr example...')
    if os.getcwd().rsplit('/', 1)[1] == 'tests':
        results = main(['','../examples/lspr'],
                        log_output=False,
                        return_results_dict=True)
    elif os.getcwd().rsplit('/', 1)[1] == 'pygbe':
        results = main(['','./examples/lspr'],
                        log_output=False,
                        return_results_dict=True)
    else:
        print("Run tests from either the main repo directory or the tests directory")

    return results
