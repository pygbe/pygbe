import os
import time
import subprocess
import datetime

from check_for_meshes import check_mesh

# tests to run
tests = ['lysozyme.py', 'molecule_dirichlet.py', 'molecule_neumann.py',
         'sphere_dirichlet.py', 'sphere_molecule_single.py',
         'sphere_molecule_stern.py', 'sphere_neumann.py',
         'twosphere_dirichlet.py', 'twosphere_neumann.py', 'two_molecules.py']

# specify CUDA device to use
CUDA_DEVICE = '0'

ENV = os.environ.copy()
ENV['CUDA_DEVICE'] = CUDA_DEVICE

check_mesh()

tic = time.time()

for test in tests:
    subprocess.call(['python', '{}'.format(test)])

toc = time.time()

print("Total runtime for convergence tests: ")
print(str(datetime.timedelta(seconds=(toc - tic))))
