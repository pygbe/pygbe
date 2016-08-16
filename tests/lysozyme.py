import pickle
import pytest

try:
    import pycuda
except ImportError:
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
def test_lysozyme(key):
    results = main(['', '../examples/lys',
                    'log_output=False',
                    'return_results_dict=True'])

    with open('lysozyme.pickle', 'r') as f:
        base_results = pickle.load(f)

    assert base_results[key] == results[key]
