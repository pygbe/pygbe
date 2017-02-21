import re
import os
import sys
import time
import numpy
import pickle
import datetime

try:
    import pycuda
except ImportError:
    ans = input('PyCUDA not found.  Regression tests will take forever.  Do you want to continue? [y/n] ')
    if ans in ['Y', 'y']:
        pass
    else:
        sys.exit()

from pygbe.main import main as pygbe

mesh = ['500', '2K', '8K', '32K']


def picklesave(test_outputs):
    with open('tests','wb') as f:
        pickle.dump(test_outputs, f, 2)

def pickleload():
    with open('tests', 'rb') as f:
        test_outputs = pickle.load(f)

    return test_outputs


