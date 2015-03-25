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

def printSummary(surf_array, field_array, param):
    Nsurf = len(surf_array)
    print '----------------------------'
    print '%i surfaces:\n'%Nsurf
    for i in range(len(surf_array)):
        N_aux = len(surf_array[i].triangle)
        rr = numpy.zeros(len(surf_array[i].tree))
        for ii in range(len(surf_array[i].tree)):
            rr[ii] = surf_array[i].tree[ii].r
        Levels = numpy.log(surf_array[i].tree[0].r/numpy.min(rr))/numpy.log(2) + 1 

        print 'Surface %i:'%i
        print '\t%i elements'%(N_aux)
        print '\tSurface type       : '+surf_array[i].surf_type
        print '\tCells              : %i'%len(surf_array[i].tree)
        print '\tTwigs              : %i'%len(surf_array[i].twig)
        print '\tLevels             : %i'%Levels
        print '\tC0 size            : %f'%surf_array[i].tree[0].r
        print '\tC0 box center      : %f, %f, %f'%(surf_array[i].tree[0].xc, surf_array[i].tree[0].yc, surf_array[i].tree[0].zc)
        print '\tTwig cell size     : %f'%(numpy.min(rr))
        print '\tRbox/theta         : %f'%(numpy.min(rr)/param.theta)
        print '\tAnalytic distance  : %f'%(numpy.average(numpy.sqrt(2*surf_array[i].Area))/param.threshold)
        print '\tElem. per sq Ang   : %f'%(1/numpy.average(surf_array[i].Area))
        print '\tMax, min, avg elem.: %s, %s, %s'%(numpy.max(surf_array[i].Area),numpy.min(surf_array[i].Area),numpy.average(surf_array[i].Area))
        print '\tTotal area         : %f'%(numpy.sum(surf_array[i].Area))

    print '----------------------------\n'

    Nfield = len(field_array)
    print '%i regions:\n'%Nfield
    for i in range(len(field_array)):
        print 'Region %i:'%i
        print '\tLaplace or Yukawa: %i'%field_array[i].LorY
        print '\tkappa            : %s'%field_array[i].kappa
        print '\tdielectric const : %s'%field_array[i].E
        print '\tNumber of charges: %i'%len(field_array[i].q)
        print '\tParent surface   : '+str(field_array[i].parent)
        print '\tChild surfaces   : '+str(field_array[i].child)

    print '----------------------------\n'

    print 'Parameters:'
    print '\tData type               : '+str(param.REAL)
    print '\tUse GPU                 : %i'%param.GPU
    print '\tP                       : %i'%param.P
    print '\tthreshold               : %.2f'%param.threshold
    print '\ttheta                   : %.2f'%param.theta
    print '\tNCRIT                   : %i'%param.NCRIT
    print '\tCUDA block size         : %i'%param.BSZ
    print '\tGauss points per element: %i'%param.K
    print '\tGauss points near singlr: %i'%param.K_fine
    print '\t1D Gauss points per side: %i'%param.Nk
    print '\tGMRES tolerance         : %s'%param.tol
    print '\tGMRES max iterations    : %i'%param.max_iter
    print '\tGMRES restart iteration : %i'%param.restart

    print '----------------------------\n'
