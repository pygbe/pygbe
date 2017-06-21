import os
import time
import subprocess
import datetime

from check_for_meshes import check_mesh

# tests to run
tests = ['sphere_lspr.py', 'sphere_multiple_lspr.py']

# specify CUDA device to use
CUDA_DEVICE = '0'

ENV = os.environ.copy()
ENV['CUDA_DEVICE'] = CUDA_DEVICE

mesh_file = 'https://zenodo.org/record/580786/files/pygbe-lspr_convergence_test_meshes.zip'
folder_name = 'lspr_convergence_test_meshes'
rename_folder = 'geometry_lspr'
size = '~3MB'

check_mesh(mesh_file, folder_name, rename_folder, size)

tic = time.time()

for test in tests:
    subprocess.call(['python', '{}'.format(test)])

toc = time.time()

print("Total runtime for convergence tests: ")
print(str(datetime.timedelta(seconds=(toc - tic))))
