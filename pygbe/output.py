"""
Prints output with the main information.
"""
import numpy 

# yapf: disable
def printSummary(surf_array, field_array, param):
    """
    Prints a summary with the main information of the run.

    Arguments
    ----------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    field_array: array, contains the Field classes of each region on the surface.
    param      : class, parameters related to the surface. 
    """
    
    Nsurf = len(surf_array)
    print 28 * '-' + '\n'
    print '%i surfaces:\n' % Nsurf
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

    print 28*'-'+'\n'

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

    print 28*'-'+'\n'

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

    print 28*'-'+'\n'
# yapf: enable
