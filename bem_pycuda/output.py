from numpy import *

def printSummary(surf_array, field_array, param):
    Nsurf = len(surf_array)
    print '----------------------------'
    print '%i surfaces:\n'%Nsurf
    for i in range(len(surf_array)):
        N_aux = len(surf_array[i].triangle)
        rr = zeros(len(surf_array[i].tree))
        for ii in range(len(surf_array[i].tree)):
            rr[ii] = surf_array[i].tree[ii].r
        Levels = log(surf_array[i].tree[0].r/min(rr))/log(2) + 1 

        print 'Surface %i:'%i
        print '\t%i elements'%(N_aux)
        print '\tCells              : %i'%len(surf_array[i].tree)
        print '\tTwigs              : %i'%len(surf_array[i].twig)
        print '\tLevels             : %i'%Levels
        print '\tC0 size            : %f'%surf_array[i].tree[0].r
        print '\tC0 box center      : %f, %f, %f'%(surf_array[i].tree[0].xc, surf_array[i].tree[0].yc, surf_array[i].tree[0].zc)
        print '\tTwig cell size     : %f'%(min(rr))
        print '\tRbox/theta         : %f'%(min(rr)/param.theta)
        print '\tAnalytic distance  : %f'%(average(sqrt(2*surf_array[i].Area))/param.threshold)
        print '\tElem. per sq Ang   : %f'%(1/average(surf_array[i].Area))

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
        print '\tNumber of charges: %i'%len(field_array[i].q)

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
