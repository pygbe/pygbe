import pickle
import pytest
# TODO change this to `import functools` for py3 port
import functools32 as functools

try:
    import pycuda
except ImportError:
    # TODO change this to `input` for py3 port
    ans = raw_input('PyCUDA not found.  Regression tests will take forever.  Do you want to continue? [y/n] ')
    if ans in ['Y', 'y']:
        pass
    else:
        sys.exit()

from pygbe.main import main

@pytest.mark.parametrize('key', ['total_elements',
                                 'E_solv_kJ',
                                 'E_coul_kcal',
                                 'E_coul_kJ',
                                 'E_solv_kcal'])
def test_PGB_mut_sensor(key):
    results = get_results()

    with open('pgbmut.pickle', 'r') as f:
        base_results = pickle.load(f)

    assert base_results[key] == results[key]

@functools.lru_cache(5)
def get_results():
    print('Generating results for 1PGBmut example...')
    results = main(['','../examples/1PGBmut_sensor'],
                    log_output=False,
                    return_results_dict=True)

    return results
