"""
It contains the function in charge of the data transfer to the GPU.
"""
import numpy

# PyCUDA libraries
try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
except:
    pass

from pygbe.classes import Event


def dataTransfer(surf_array, field_array, ind, param, kernel):
    """
    It manages the data transfer to the GPU.

    Arguments
    ----------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    field_array: array, contains the Field classes of each region on the surface.
    ind        : class, it contains the indices related to the treecode
                        computation.
    param      : class, parameters related to the surface.
    kernel: pycuda source module.
    """

    REAL = param.REAL
    Nsurf = len(surf_array)
    for s in range(Nsurf):
        surf_array[s].xiDev = gpuarray.to_gpu(surf_array[s].xiSort.astype(
            REAL))
        surf_array[s].yiDev = gpuarray.to_gpu(surf_array[s].yiSort.astype(
            REAL))
        surf_array[s].ziDev = gpuarray.to_gpu(surf_array[s].ziSort.astype(
            REAL))
        surf_array[s].xjDev = gpuarray.to_gpu(surf_array[s].xjSort.astype(
            REAL))
        surf_array[s].yjDev = gpuarray.to_gpu(surf_array[s].yjSort.astype(
            REAL))
        surf_array[s].zjDev = gpuarray.to_gpu(surf_array[s].zjSort.astype(
            REAL))
        surf_array[s].AreaDev = gpuarray.to_gpu(surf_array[s].AreaSort.astype(
            REAL))
        surf_array[s].sglInt_intDev = gpuarray.to_gpu(surf_array[
            s].sglInt_intSort.astype(REAL))
        surf_array[s].sglInt_extDev = gpuarray.to_gpu(surf_array[
            s].sglInt_extSort.astype(REAL))
        surf_array[s].vertexDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].vertex[surf_array[s].triangleSort]).astype(REAL))
        surf_array[s].xcDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].xcSort.astype(REAL)))
        surf_array[s].ycDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].ycSort.astype(REAL)))
        surf_array[s].zcDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].zcSort.astype(REAL)))

        #       Avoid transferring size 1 arrays to GPU (some systems crash)
        Nbuff = 5
        if len(surf_array[s].sizeTarget) < Nbuff:
            sizeTarget_buffer = numpy.zeros(Nbuff, dtype=numpy.int32)
            sizeTarget_buffer[:len(surf_array[s].sizeTarget)] = surf_array[
                s].sizeTarget[:]
            surf_array[s].sizeTarDev = gpuarray.to_gpu(sizeTarget_buffer)
        else:
            surf_array[s].sizeTarDev = gpuarray.to_gpu(surf_array[
                s].sizeTarget.astype(numpy.int32))

    #        surf_array[s].sizeTarDev = gpuarray.to_gpu(surf_array[s].sizeTarget.astype(numpy.int32))
        surf_array[s].offSrcDev = gpuarray.to_gpu(surf_array[
            s].offsetSource.astype(numpy.int32))
        surf_array[s].offTwgDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].offsetTwigs.astype(numpy.int32)))
        surf_array[s].offMltDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].offsetMlt.astype(numpy.int32)))
        surf_array[s].M2P_lstDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].M2P_list.astype(numpy.int32)))
        surf_array[s].P2P_lstDev = gpuarray.to_gpu(numpy.ravel(surf_array[
            s].P2P_list.astype(numpy.int32)))
        surf_array[s].xkDev = gpuarray.to_gpu(surf_array[s].xk.astype(REAL))
        surf_array[s].wkDev = gpuarray.to_gpu(surf_array[s].wk.astype(REAL))
        surf_array[s].XskDev = gpuarray.to_gpu(surf_array[s].Xsk.astype(REAL))
        surf_array[s].WskDev = gpuarray.to_gpu(surf_array[s].Wsk.astype(REAL))
        surf_array[s].kDev = gpuarray.to_gpu((surf_array[s].sortSource %
                                              param.K).astype(numpy.int32))

    ind.indexDev = gpuarray.to_gpu(ind.index_large.astype(numpy.int32))

    Nfield = len(field_array)
    for f in range(Nfield):
        if len(field_array[f].xq) > 0:
            field_array[f].xq_gpu = gpuarray.to_gpu(field_array[
                f].xq[:, 0].astype(REAL))
            field_array[f].yq_gpu = gpuarray.to_gpu(field_array[
                f].xq[:, 1].astype(REAL))
            field_array[f].zq_gpu = gpuarray.to_gpu(field_array[
                f].xq[:, 2].astype(REAL))
            field_array[f].q_gpu = gpuarray.to_gpu(field_array[f].q.astype(
                REAL))
