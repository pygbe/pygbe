'''
  Copyright (C) 2013 by Christopher Cooper, Lorena Barba

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
'''

import os
import time

print '*****REGRESSION TESTS*****'
print '=========================='

# Timestamp
timestamp = time.localtime()
DEVICE = 2
print 'Run started on:'
print '\tDate: %i/%i/%i'%(timestamp.tm_year,timestamp.tm_mon,timestamp.tm_mday)
print '\tTime: %i:%i:%i'%(timestamp.tm_hour,timestamp.tm_min,timestamp.tm_sec)

print '\nISOLATED CHARGED SPHERES'
print '-----------------------'
print '\nDirichlet surface'
print '-----------------'
comm = 'CUDA_DEVICE=%i python regression_tests/sphere_dirichlet.py'%DEVICE
os.system(comm)

print '\nNeumann surface'
print '---------------'
comm = 'CUDA_DEVICE=%i python regression_tests/sphere_neumann.py'%DEVICE
os.system(comm)
print '----------------------------------------------'

print '\nISOLATED SPHERICAL MOLECULE'
print '---------------------------'
print '\nNo stern layer'
print '--------------'
comm = 'CUDA_DEVICE=%i python regression_tests/sphere_molecule_single.py'%DEVICE
os.system(comm)
print '\nWith Stern layer'
print '----------------'
comm = 'CUDA_DEVICE=%i python regression_tests/sphere_molecule_stern.py'%DEVICE
os.system(comm)
print '----------------------------------------------'

print '\nTWO CHARGED SPHERES'
print '-------------------'
print '\nBoth Dirichlet surfaces'
print '-----------------------'
comm = 'CUDA_DEVICE=%i python regression_tests/twosphere_dirichlet.py'%DEVICE
os.system(comm)
print '\nBoth Neumann surfaces'
print '---------------------'
comm = 'CUDA_DEVICE=%i python regression_tests/twosphere_neumann.py'%DEVICE
os.system(comm)
print '----------------------------------------------'

print '\nTWO MOLECULES'
print '-------------'
comm = 'CUDA_DEVICE=%i python regression_tests/two_molecules.py'%DEVICE
os.system(comm)
print '----------------------------------------------'

print '\nCHARGED SPHERE WITH SPHERICAL MOLECULE'
print '--------------------------------------'
print '\nWith Dirichlet surface'
print '----------------------'
comm = 'CUDA_DEVICE=%i python regression_tests/molecule_dirichlet.py'%DEVICE
os.system(comm)
print '\nWith Neumann surface'
print '--------------------'
comm = 'CUDA_DEVICE=%i python regression_tests/molecule_neumann.py'%DEVICE
os.system(comm)
print '----------------------------------------------'

print '\nLYSOZYME TESTS'
print '--------------'
comm = 'CUDA_DEVICE=%i python regression_tests/lysozyme.py'%DEVICE
os.system(comm)

print '\nPARAMETER FILES'
print '---------------'
print '\nSphere parameters'
print '-----------------'
for line in file('regression_tests/input_files/sphere_fine.param'):
    print line

#print '\nSphere standard parameters'
#print '-----------------'
#for line in file('regression_tests/input_files/sphere_standard.param'):
#    print line

print '\nLysozyme parameters'
print '-------------------'
for line in file('regression_tests/input_files/lys.param'):
    print line

