#!/usr/bin/env python
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

import numpy
from math import pi
from scipy.misc import factorial
import time
from datetime import datetime
import os
import sys
from argparse import ArgumentParser

# Import self made modules
from gmres import gmres_solver
from projection import get_phir
from classes import (surfaces, timings, parameters, index_constant,
                     fill_surface, initializeSurf, initializeField,
                     dataTransfer, fill_phi)
from output import printSummary
from matrixfree import (generateRHS, generateRHS_gpu, calculateEsolv,
                        coulombEnergy, calculateEsurf)

from util.readData import readVertex, readTriangle, readpqr, readParameters
from util.an_solution import an_P, two_sphere
from util.which import whichgen

from tree.FMMutils import computeIndices, precomputeTerms, generateList

try:
    from tree.cuda_kernels import kernels
except:
    pass

#courtesy of http://stackoverflow.com/a/5916874
class Logger(object):
    """
    Allow writing both to STDOUT on screen and sending text to file
    in conjunction with the command
    `sys.stdout = Logger("desired_log_file.txt")`
    """
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

def read_inputs():
    """
    Parse command-line arguments to determine which config and param files to run
    Assumes that in the absence of specific command line arguments that pygbe
    problem folder resembles the following structure

    lys
    |- lys.param
    |- lys.config
    |- built_parse.pqr
    |- geometry/Lys1.face
    |- geometry/Lys1.vert
    |- output/
    """
    parser = ArgumentParser(description='Manage PyGBe command line arguments')
    parser.add_argument('problem_folder', type=str,
                        help="Path to folder containing problem files")
    parser.add_argument('-c', '--config', dest='config', type=str, default=None,
                        help="Path to problem config file")
    parser.add_argument('-p', '--param', dest='param', type=str, default=None,
                        help="Path to problem param file")
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='output', help="Output folder")

    return parser.parse_args()

def check_file_exists(filename):
    """Try to open the file `filename` and return True if it's valid """
    return os.path.exists(filename)

def find_config_files(cliargs):
    """
    Check that .config and .param files exist and can be opened.
    If either file isn't found, PyGBe exits (and should print which
    file was not found).  Otherwise return the path to the config and
    param files

    Parameters
    ----------
    cliargs: parser
        parser containing cli arguments passed to PyGBe

    Returns
    -------
    cliargs.config: string
        path to config file
    cliargs.param: string
        path to param file
    """

    prob_path = cliargs.problem_folder
    full_path = os.getcwd() + '/' + prob_path
    #if user gave us an absolute path, use that
    if not os.path.isdir(full_path):
        full_path = os.path.expanduser(prob_path)
    os.environ['PYGBE_PROBLEM_FOLDER'] = full_path
    #If user tries `pygbe lys` then `split` will fail
    #But if user tries `pygbe examples/lys` then the split
    #is required to grab the right name
    try:
        prob_name = (prob_path.split('/')[-1] if prob_path[-1] != '/'
                     else prob_path.split('/')[-2])
    except AttributeError:
        prob_name = prob_path

    if cliargs.config is None:
        cliargs.config = os.path.join(full_path, prob_name+'.config')
    if cliargs.param is None:
        cliargs.param = os.path.join(full_path, prob_name+'.param')

    for req_file in [cliargs.config, cliargs.param]:
        if not check_file_exists(req_file):
            sys.exit('Did not find expected config files')

    return cliargs.config, cliargs.param

def check_for_nvcc():
    '''Check system PATH for nvcc, exit if not found'''
    try:
        whichgen('nvcc').next()
    except StopIteration:
        print("Could not find `nvcc` on your PATH.  Is cuda installed?  PyGBe will continue to run but will run significantly slower.  For optimal performance, add `nvcc` to your PATH")


def main(log_output=True):

    check_for_nvcc()

    args = read_inputs()
    configFile, paramfile = find_config_files(args)
    full_path = os.environ.get('PYGBE_PROBLEM_FOLDER') + '/'

    #try to expand ~ if present in output path
    args.output = os.path.expanduser(args.output)
    #if output path is absolute, use that, otherwise prepend
    #problem path
    if not os.path.isdir(args.output):
        output_dir = os.path.join(full_path, args.output)
    else:
        output_dir = args.output
    #create output directory if it doesn't already exist
    try:
        os.makedirs(output_dir)
    except OSError:
        pass


    timestamp = time.localtime()
    outputfname = '{:%Y-%m-%d-%H%M%S}-output.log'.format(datetime.now())
    if log_output:
        sys.stdout = Logger(os.path.join(output_dir, outputfname))
    ### Time stamp
    print 'Run started on:'
    print '\tDate: %i/%i/%i'%(timestamp.tm_year,timestamp.tm_mon,timestamp.tm_mday)
    print '\tTime: %i:%i:%i'%(timestamp.tm_hour,timestamp.tm_min,timestamp.tm_sec)
    TIC = time.time()

    ### Read parameters
    param = parameters()
    precision = readParameters(param,paramfile)

    param.Nm            = (param.P+1)*(param.P+2)*(param.P+3)/6     # Number of terms in Taylor expansion
    param.BlocksPerTwig = int(numpy.ceil(param.NCRIT/float(param.BSZ)))   # CUDA blocks that fit per twig

    ### Generate array of fields
    field_array = initializeField(configFile, param)

    ### Generate array of surfaces and read in elements
    surf_array = initializeSurf(field_array, configFile, param)

    ### Fill surface class
    time_sort = 0.
    for i in range(len(surf_array)):
        time_sort += fill_surface(surf_array[i], param)

    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    #ss=surf_array[0]
    for i in range(1):
        ss = surf_array[i]
        ax.scatter(ss.xi,ss.yi,ss.zi,c='b',marker='o')
        ax.scatter(ss.xi+ss.normal[:,0], ss.yi+ss.normal[:,1], ss.zi+ss.normal[:,2],c='r', marker='o')
    plt.show()
    quit()
    '''

    ### Output setup summary
    param.N   = 0
    param.Neq = 0
    for s in surf_array:
        N_aux = len(s.triangle)
        param.N += N_aux
        if s.surf_type == 'dirichlet_surface' or s.surf_type == 'neumann_surface' or s.surf_type == 'asc_surface':
            param.Neq += N_aux
        else:
            param.Neq += 2*N_aux
    print '\nTotal elements : %i'%param.N
    print 'Total equations: %i'%param.Neq

    printSummary(surf_array, field_array, param)

    ### Precomputation
    ind0 = index_constant()
    computeIndices(param.P, ind0)
    precomputeTerms(param.P, ind0)

    ### Load CUDA code
    if param.GPU==1:
        kernel = kernels(param.BSZ, param.Nm, param.K_fine, param.P, precision)
    else:
        kernel = 1

    ### Generate interaction list
    print 'Generate interaction list'
    tic = time.time()
    generateList(surf_array, field_array, param)
    toc = time.time()
    list_time = toc-tic

    ### Transfer data to GPU
    print 'Transfer data to GPU'
    tic = time.time()
    if param.GPU==1:
        dataTransfer(surf_array, field_array, ind0, param, kernel)
    toc = time.time()
    transfer_time = toc-tic

    timing = timings()

    ### Generate RHS
    print 'Generate RHS'
    tic = time.time()
    if param.GPU==0:
        F = generateRHS(field_array, surf_array, param, kernel, timing, ind0)
    elif param.GPU==1:
        F = generateRHS_gpu(field_array, surf_array, param, kernel, timing, ind0)
    toc = time.time()
    rhs_time = toc-tic

#    numpy.savetxt(os.path.join(output_dir,'RHS.txt'),F)

    setup_time = toc-TIC
    print 'List time          : %fs'%list_time
    print 'Data transfer time : %fs'%transfer_time
    print 'RHS generation time: %fs'%rhs_time
    print '------------------------------'
    print 'Total setup time   : %fs\n'%setup_time

    tic = time.time()

    ### Solve
    print 'Solve'
    phi = numpy.zeros(param.Neq)
    phi = gmres_solver(surf_array, field_array, phi, F, param, ind0, timing, kernel) 
    toc = time.time()
    solve_time = toc-tic
    print 'Solve time        : %fs'%solve_time
    phifname = '{:%Y-%m-%d-%H%M%S}-phi.txt'.format(datetime.now())
    numpy.savetxt(os.path.join(output_dir, phifname),phi)
    #phi = loadtxt('phi.txt')

    # Put result phi in corresponding surfaces
    fill_phi(phi, surf_array)

    ### Calculate solvation energy
    print '\nCalculate Esolv'
    tic = time.time()
    E_solv = calculateEsolv(surf_array, field_array, param, kernel)
    toc = time.time()
    print 'Time Esolv: %fs'%(toc-tic)
    ii = -1
    for f in param.E_field:
        parent_type = surf_array[field_array[f].parent[0]].surf_type
        if parent_type != 'dirichlet_surface' and parent_type != 'neumann_surface':
            ii += 1
            print 'Region %i: Esolv = %f kcal/mol = %f kJ/mol'%(f, E_solv[ii], E_solv[ii]*4.184)

    ### Calculate surface energy
    print '\nCalculate Esurf'
    tic = time.time()
    E_surf = calculateEsurf(surf_array, field_array, param, kernel)
    toc = time.time()
    ii = -1
    for f in param.E_field:
        parent_type = surf_array[field_array[f].parent[0]].surf_type
        if parent_type == 'dirichlet_surface' or parent_type == 'neumann_surface':
            ii += 1
            print 'Region %i: Esurf = %f kcal/mol = %f kJ/mol'%(f, E_surf[ii], E_surf[ii]*4.184)
    print 'Time Esurf: %fs'%(toc-tic)

    ### Calculate Coulombic interaction
    print '\nCalculate Ecoul'
    tic = time.time()
    i = -1
    E_coul = []
    for f in field_array:
        i += 1
        if f.coulomb == 1:
            print 'Calculate Coulomb energy for region %i'%i
            E_coul.append(coulombEnergy(f, param))
            print 'Region %i: Ecoul = %f kcal/mol = %f kJ/mol'%(i,E_coul[-1],E_coul[-1]*4.184)
    toc = time.time()
    print 'Time Ecoul: %fs'%(toc-tic)

    ### Output summary
    print '\n--------------------------------'
    print 'Totals:'
    print 'Esolv = %f kcal/mol'%sum(E_solv)
    print 'Esurf = %f kcal/mol'%sum(E_surf)
    print 'Ecoul = %f kcal/mol'%sum(E_coul)
    print '\nTime = %f s'%(toc-TIC)

    # Analytic solution
    '''
    # two spheres
    R1 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0])
    dist = norm(field_array[2].xq[0]-field_array[1].xq[0])
    E_1 = field_array[1].E
    E_2 = field_array[0].E
    E_an,E1an,E2an = two_sphere(R1, dist, field_array[0].kappa, E_1, E_2, field_array[1].q[0])
    JtoCal = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(JtoCal*param.E_0)
    E_an *= C0/(4*pi)
    E1an *= C0/(4*pi)
    E2an *= C0/(4*pi)
    print '\n E_solv = %s kcal/mol, Analytical solution = %f kcal/mol, Error: %s'%(E_solv, E2an, abs(E_solv-E2an)/abs(E2an))
    print '\n E_solv = %s kJ/mol, Analytical solution = %f kJ/mol, Error: %s'%(E_solv*JtoCal, E2an*JtoCal, abs(E_solv-E2an)/abs(E2an))

    # sphere with stern layer
    K_sph = 20 # Number of terms in spherical harmonic expansion
    #E_1 = field_array[2].E # stern
    E_1 = field_array[1].E # no stern
    E_2 = field_array[0].E
    R1 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0])
    #R2 = norm(surf_array[1].vertex[surf_array[0].triangle[0]][0]) # stern
    R2 = norm(surf_array[0].vertex[surf_array[0].triangle[0]][0]) # no stern
    #q = field_array[2].q # stern
    q = field_array[1].q # no stern
    #xq = field_array[2].xq # stern
    xq = field_array[1].xq # no stern
    xq += 1e-12
    print q, xq, E_1, E_2, R1, field_array[0].kappa, R2, K_sph
    phi_P = an_P(q, xq, E_1, E_2, R1, field_array[0].kappa, R2, K_sph)
    JtoCal = 4.184
    E_P = 0.5*param.qe**2*sum(q*phi_P)*param.Na*1e7/JtoCal
    print '\n E_solv = %s, Legendre polynomial sol = %f, Error: %s'%(E_solv, E_P, abs(E_solv-E_P)/abs(E_P))
    '''


if __name__ == "__main__":
    sys.exit(main(sys.argv))
