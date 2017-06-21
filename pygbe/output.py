"""
Prints output with the main information.
"""
import numpy

# yapf: disable
def print_summary(surf_array, field_array, param, results_dict):
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
    print(28 * '-' + '\n')
    print('{} surfaces:\n'.format(Nsurf))
    for i in range(len(surf_array)):
        N_aux = len(surf_array[i].triangle)
        rr = numpy.zeros(len(surf_array[i].tree))
        for ii in range(len(surf_array[i].tree)):
            rr[ii] = surf_array[i].tree[ii].r
        Levels = int(numpy.log(surf_array[i].tree[0].r/numpy.min(rr))/numpy.log(2) + 1)

        try:
            key = 'elem_sq_ang_surf{}'.format(i)
            results_dict[key] = [1/numpy.average(surf_array[i].area)]
        except IndexError:
            pass

        print('Surface {}:'.format(i))
        print('\t{} elements'.format(N_aux))
        print('\tSurface type         : {}'.format(surf_array[i].surf_type))
        print('\tCells                : {}'.format(len(surf_array[i].tree)))
        print('\tTwigs                : {}'.format(len(surf_array[i].twig)))
        print('\tLevels               : {}'.format(Levels))
        print('\tC0 size              : {}'.format(surf_array[i].tree[0].r))
        print('\tC0 box center        : {}, {}, {}'.format(surf_array[i].tree[0].xc,
                                                         surf_array[i].tree[0].yc,
                                                         surf_array[i].tree[0].zc))
        print('\tTwig cell size       : {}'.format(numpy.min(rr)))
        print('\tRbox/theta           : {}'.format(numpy.min(rr)/param.theta))
        print('\tAnalytic distance    : {}'.format(numpy.average(numpy.sqrt(2*surf_array[i].area))/param.threshold))
        print('\tElem. per Ang^2      : {}'.format(1/numpy.average(surf_array[i].area)))
        print('\tMax, min, avg elem.  : {}, {}, {}'.format(numpy.max(surf_array[i].area),
                                                         numpy.min(surf_array[i].area),
                                                         numpy.average(surf_array[i].area)))
        print('\tTotal area           : {}'.format(numpy.sum(surf_array[i].area)))

    print(30*'-'+'\n')

    Nfield = len(field_array)
    print('{} regions:\n'.format(Nfield))
    for i in range(len(field_array)):
        print('Region {}:'.format(i))
        print('\tLaplace or Yukawa: {}'.format(field_array[i].LorY))
        print('\tkappa            : {}'.format(field_array[i].kappa))
        print('\tdielectric const : {}'.format(field_array[i].E))
        print('\tNumber of charges: {}'.format(len(field_array[i].q)))
        print('\tParent surface   : {}'.format(field_array[i].parent))
        print('\tChild surfaces   : {}'.format(field_array[i].child))

    print(30*'-'+'\n')

    print('Parameters:')
    print('\tData type               : {}'.format(param.REAL))
    print('\tUse GPU                 : {}'.format(param.GPU))
    print('\tP                       : {}'.format(param.P))
    print('\tthreshold               : {:.2f}'.format(param.threshold))
    print('\ttheta                   : {:.2f}'.format(param.theta))
    print('\tNCRIT                   : {}'.format(param.NCRIT))
    print('\tCUDA block size         : {}'.format(param.BSZ))
    print('\tGauss points per element: {}'.format(param.K))
    print('\tGauss points near singlr: {}'.format(param.K_fine))
    print('\t1D Gauss points per side: {}'.format(param.Nk))
    print('\tGMRES tolerance         : {}'.format(param.tol))
    print('\tGMRES max iterations    : {}'.format(param.max_iter))
    print('\tGMRES restart iteration : {}'.format(param.restart))

    print(28*'-'+'\n')

    return results_dict
# yapf: enable
