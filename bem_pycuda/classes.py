from numpy import *
import sys
sys.path.append('tree')
from FMMutils import *
sys.path.append('../util')
from semi_analytical    import *
from triangulation      import *
from readData           import readVertex, readTriangle, readpqr, readcrd, readFields, readSurf

# PyCUDA libraries
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

class surfaces():
    def __init__(self):
        self.triangle = []  # indices to triangle vertices
        self.vertex   = []  # position of vertices
        self.XinV     = []  # weights input for single layer potential
        self.XinK     = []  # weights input for double layer potential
        self.Xout_int = []  # output vector of interior operators
        self.Xout_ext = []  # output vector of exterior operators
        self.xi       = []  # x component of center
        self.yi       = []  # y component of center
        self.zi       = []  # z component of center
        self.xj       = []  # x component of gauss nodes
        self.yj       = []  # y component of gauss nodes
        self.zj       = []  # z component of gauss nodes
        self.Area     = []  # Area of triangles
        self.normal   = []  # normal of triangles
        self.xk       = []  # position of gauss points on edges
        self.wk       = []  # weight of gauss points on edges
        self.tree     = []  # tree structure
        self.twig     = []  # tree twigs
        self.xiSort   = []  # sorted x component of center
        self.yiSort   = []  # sorted y component of center
        self.ziSort   = []  # sorted z component of center
        self.xjSort   = []  # sorted x component of gauss nodes
        self.yjSort   = []  # sorted y component of gauss nodes
        self.zjSort   = []  # sorted z component of gauss nodes
        self.xcSort   = []  # sorted x component box centers according to M2P_list array
        self.ycSort   = []  # sorted y component box centers according to M2P_list array
        self.zcSort   = []  # sorted z component box centers according to M2P_list array
        self.AreaSort = []  # sorted array of areas
        self.unsort       = []  # array of indices to unsort targets
        self.triangleSort = [] # sorted array of triangles
        self.sortTarget   = []  # array of indices to sort targets
        self.sortSource   = []  # array of indices to sort sources
        self.offsetSource = [] # array with offsets to sorted source array
        self.offsetTarget = [] # array with offsets to sorted target array
        self.sizeTarget   = [] # array with number of targets pero twig
        self.offsetTwigs  = [] # offset to twig in P2P list array
        self.P2P_list     = [] # pointers to twigs for P2P interaction list
        self.offsetMlt    = [] # offset to multipoles in M2P list array
        self.M2P_list     = [] # pointers to boxes for M2P interaction list
        self.Precond      = [] # Sparse representation of preconditioner for self interaction block
        self.E_hat        = 0  # ratio of Ein/Eout
        self.kappa_in     = 0  # kappa inside surface
        self.kappa_out    = 0  # kappa inside surface

        # Device data
        self.xiDev      = []
        self.yiDev      = []
        self.ziDev      = []
        self.xjDev      = []
        self.yjDev      = []
        self.zjDev      = []
        self.xcDev      = []
        self.ycDev      = []
        self.zcDev      = []
        self.AreaDev    = []
        self.vertexDev  = []
        self.sizeTarDev = []
        self.offSrcDev  = []
        self.offMltDev  = []
        self.offTwgDev  = []
        self.M2P_lstDev = []
        self.P2P_lstDev = []
        self.xkDev      = []
        self.wkDev      = []
        self.kDev       = []

class fields():
    def __init__(self):
        self.parent = []    # Pointer to "parent" surface
        self.child  = []    # Pointer to "children" surfaces
        self.LorY   = []    # 1: Laplace, 2: Yukawa
        self.kappa  = []    # inverse of Debye length
        self.E      = []    # dielectric constant
        self.xq     = []    # position of charges
        self.q      = []    # value of charges

        # Device data
        self.xq_gpu = []    # x position of charges on gpu
        self.yq_gpu = []    # y position of charges on gpu
        self.zq_gpu = []    # z position of charges on gpu
        self.q_gpu  = []    # value of charges on gpu

class timings():
    def __init__(self):
        self.time_an    = 0.
        self.time_P2P   = 0.
        self.time_P2M   = 0.
        self.time_M2M   = 0.
        self.time_M2P   = 0.
        self.time_trans = 0.
        self.time_sort  = 0.
        self.time_mass  = 0.
        self.AI_int     = 0


class parameters():
    def __init__(self):
        self.kappa         = 0.              # inverse of Debye length
        self.restart       = 0               # Restart of GMRES
        self.tol           = 0.              # Tolerance of GMRES
        self.max_iter      = 0               # Max number of GMRES iterations
        self.P             = 0               # Order of Taylor expansion
        self.eps           = 0               # Epsilon machine
        self.Nm            = 0               # Number of terms in Taylor expansion
        self.NCRIT         = 0               # Max number of targets per twig box
        self.theta         = 0.              # MAC criterion for treecode
        self.K             = 0               # Number of Gauss points per element
        self.threshold     = 0.              # L/d criterion for semi-analytic intergrals
        self.Nk            = 0               # Gauss points per side for semi-analytical integrals
        self.BSZ           = 0               # CUDA block size
        self.Nround        = 0               # Max size of sorted target array
        self.BlocksPerTwig = 0               # Number of CUDA blocks that fit per tree twig
        self.N             = 0               # Total number of elements
        self.qe            = 1.60217646e-19  # Charge of an electron
        self.Na            = 6.0221415e23    # Avogadro's number
        self.E_0           = 8.854187818e-12 # Vacuum dielectric constant
        self.REAL          = 0               # Data type
        self.GPU           = -1              # =1: with GPU, =0: no GPU

class index_constant():
    def __init__(self):
        self.II = []
        self.JJ = []
        self.KK = []
        self.index       = []
        self.index_small = []
        self.index_large = []
        self.index_ptr   = []
        self.combII = []
        self.combJJ = []
        self.combKK = []
        self.IImii  = []
        self.JJmjj  = []
        self.KKmkk  = []

        # Device data
        self.indexDev = []


def getGaussPoints(y,triangle, n):
    # y         : vertices
    # triangle  : array with indices for corresponding triangles
    # n         : Gauss points per element

    N  = len(triangle) # Number of triangles
    xi = zeros((N*n,3))
    if n==1:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[i,:] = dot(M, 1/3.*ones(3))

    if n==3:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[n*i,:] = dot(M, array([0.5, 0.5, 0.]))
            xi[n*i+1,:] = dot(M, array([0., 0.5, 0.5]))
            xi[n*i+2,:] = dot(M, array([0.5, 0., 0.5]))

    if n==4:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[n*i,:] = dot(M, array([1/3., 1/3., 1/3.]))
            xi[n*i+1,:] = dot(M, array([3/5., 1/5., 1/5.]))
            xi[n*i+2,:] = dot(M, array([1/5., 3/5., 1/5.]))
            xi[n*i+3,:] = dot(M, array([1/5., 1/5., 3/5.]))

    if n==7:
        for i in range(N):
            M = transpose(y[triangle[i]])
            xi[n*i+0,:] = dot(M, array([1/3.,1/3.,1/3.]))
            xi[n*i+1,:] = dot(M, array([.79742699,.10128651,.10128651]))
            xi[n*i+2,:] = dot(M, array([.10128651,.79742699,.10128651]))
            xi[n*i+3,:] = dot(M, array([.10128651,.10128651,.79742699]))
            xi[n*i+4,:] = dot(M, array([.05971587,.47014206,.47014206]))
            xi[n*i+5,:] = dot(M, array([.47014206,.05971587,.47014206]))
            xi[n*i+6,:] = dot(M, array([.47014206,.47014206,.05971587]))

    return xi[:,0], xi[:,1], xi[:,2]

def fill_surface(surf,param):

    N  = len(surf.triangle)
    Nj = N*param.K
    # Calculate centers
    surf.xi = zeros(N)
    surf.yi = zeros(N)
    surf.zi = zeros(N)
    for i in range(N):
        surf.xi[i] = average(surf.vertex[surf.triangle[i],:,0])
        surf.yi[i] = average(surf.vertex[surf.triangle[i],:,1])
        surf.zi[i] = average(surf.vertex[surf.triangle[i],:,2])

    surf.normal = zeros((N,3))
    surf.Area = zeros(N)

    for i in range(N):
        L0 = surf.vertex[surf.triangle[i,1]] - surf.vertex[surf.triangle[i,0]]
        L2 = surf.vertex[surf.triangle[i,0]] - surf.vertex[surf.triangle[i,2]]
        surf.normal[i,:] = cross(L0,L2)
        surf.Area[i] = linalg.norm(surf.normal[i,:])/2
        surf.normal[i,:] = surf.normal[i,:]/(2*surf.Area[i])

    '''
    n1 = surf.xi + surf.normal[:,0]
    n2 = surf.yi + surf.normal[:,1]
    n3 = surf.zi + surf.normal[:,2]

    rtest = sqrt(n1*n1+n2*n2+n3*n3)
    counter = 0
    for i in range(len(rtest)):
        if rtest[i]<R:
            counter += 1
    print 'wrong normal %i times'%counter
    '''


    # Set Gauss points (sources)
    surf.xj,surf.yj,surf.zj = getGaussPoints(surf.vertex,surf.triangle,param.K)

    x_center = zeros(3)
    x_center[0] = average(surf.xi).astype(param.REAL)
    x_center[1] = average(surf.yi).astype(param.REAL)
    x_center[2] = average(surf.zi).astype(param.REAL)
    dist = sqrt((surf.xi-x_center[0])**2+(surf.yi-x_center[1])**2+(surf.zi-x_center[2])**2)
    R_C0 = max(dist)

    # Generate tree, compute indices and precompute terms for M2M
    surf.tree = generateTree(surf.xi,surf.yi,surf.zi,param.NCRIT,param.Nm,N,R_C0,x_center)
    C = 0
    surf.twig = findTwigs(surf.tree, C, surf.twig, param.NCRIT)

    addSources3(surf.tree, surf.twig, param.K)
#    addSources(surf.xj,surf.yj,surf.zj,surf.tree,surf.twig)
#    for j in range(Nj):
#        C = 0
#        addSources2(surf.xj,surf.yj,surf.zj,j,surf.tree,C,param.NCRIT)

    tic = time.time()
    sortPoints(surf, surf.tree, surf.twig, param)
    toc = time.time()
    time_sort = toc - tic

    surf.xk,surf.wk = GQ_1D(param.Nk)

    # Generate preconditioner
    # Will use block-diagonal preconditioner (AltmanBardhanWhiteTidor2008)
    dX11 = zeros(N) # Top left block
    dX12 = zeros(N) # Top right block
    dX21 = zeros(N) # Bottom left block
    dX22 = zeros(N) # Bottom right block
    surf.Precond = zeros((4,N))  # Stores the inverse of the block diagonal (also a tridiag matrix)
                            # Order: Top left, top right, bott left, bott right    
    for i in range(N):
        panel = surf.vertex[surf.triangle[i]]
        center = array([surf.xi[i],surf.yi[i],surf.zi[i]])
        same = array([1], dtype=int32)
        Aaux = zeros(1, dtype=param.REAL) # Top left
        Baux = zeros(1, dtype=param.REAL) # Top right
        Caux = zeros(1, dtype=param.REAL) # Bottom left
        Daux = zeros(1, dtype=param.REAL) # Bottom right

        SA_wrap_arr(ravel(panel), center, Daux, Caux, Baux, Aaux, surf.kappa_out, same, surf.xk, surf.wk)
        dX11[i] = Aaux
        dX12[i] = -Baux
        dX21[i] = -Caux
        dX22[i] = surf.E_hat*Daux

    d_aux = 1/(dX22-dX21*dX12/dX11)
    surf.Precond[0,:] = 1/dX11 + 1/dX11*dX12*d_aux*dX21/dX11
    surf.Precond[1,:] = -1/dX11*dX12*d_aux
    surf.Precond[2,:] = -d_aux*dX21/dX11
    surf.Precond[3,:] = d_aux

    return time_sort


def initializeField(filename, param):
    
    LorY, pot, E, kappa, charges, qfile, Nparent, parent, Nchild, child = readFields(filename)

    Nfield = len(LorY)
    field_array = []
    Nchild_aux = 0
    for i in range(Nfield):
        if int(pot[i])==1:
            param.E_field = i                                       # This field is where the energy will be calculated
        field_aux = fields()
        field_aux.LorY  = int(LorY[i])                              # Laplace of Yukawa
        field_aux.E     = param.REAL(E[i])                          # Dielectric constant
        field_aux.kappa = param.REAL(kappa[i])                      # inverse Debye length
        if int(charges[i])==1:                                      # if there are charges
#            xq,q,Nq = readcrd(qfile[i], param.REAL)                 # read charges
            xq,q,Nq = readpqr(qfile[i], param.REAL)                 # read charges
            field_aux.xq = xq                                       # charges positions
            field_aux.q = q                                         # charges values
        if int(Nparent[i])==1:                                      # if it is an enclosed region
            field_aux.parent.append(int(parent[i]))                 # pointer to parent surface (enclosing surface)
        if int(Nchild[i])>0:                                        # if there are enclosed regions inside
            for j in range(int(Nchild[i])):
                field_aux.child.append(int(child[Nchild_aux+j]))    # Loop over children to get pointers
            Nchild_aux += int(Nchild[i])-1                             # Point to child for next surface
        Nchild_aux += 1

        field_array.append(field_aux)
    return field_array

    
def zeroAreas(s, triangle_raw, Area_null):
    for i in range(len(triangle_raw)):
        L0 = s.vertex[triangle_raw[i,1]] - s.vertex[triangle_raw[i,0]]
        L2 = s.vertex[triangle_raw[i,0]] - s.vertex[triangle_raw[i,2]]
        normal_aux = cross(L0,L2)
        Area_aux = linalg.norm(normal_aux)/2
        if Area_aux<1e-10:
            Area_null.append(i)
    return Area_null 

def initializeSurf(field_array, filename, param):

    surf_array = []

    files = readSurf(filename)      # Read filenames for surfaces
    Nsurf = len(files)

    for i in range(Nsurf):
        print 'Reading surface %i'%i
        print 'File: '+str(files[i])
        s = surfaces()
        Area_null = []
        s.vertex = readVertex(files[i]+'.vert', param.REAL)
        triangle_raw = readTriangle(files[i]+'.face')
        Area_null = zeroAreas(s, triangle_raw, Area_null)
        s.triangle = delete(triangle_raw, Area_null, 0)
        print 'Removed areas=0: %i'%len(Area_null)

        # Look for regions inside/outside
        for j in range(Nsurf+1):
            if len(field_array[j].parent)>0:
                if field_array[j].parent[0]==i:                 # Inside region
                    s.kappa_in = field_array[j].kappa
                    Ein = field_array[j].E
            if len(field_array[j].child)>0:
                if i in field_array[j].child:                # Outside region
                    s.kappa_out = field_array[j].kappa
                    Eout = field_array[j].E
        s.E_hat = Ein/Eout

        surf_array.append(s)

    return surf_array

def dataTransfer(surf_array, field_array, ind, param):

    REAL = param.REAL
    Nsurf = len(surf_array)
    for s in range(Nsurf):
        surf_array[s].xiDev      = gpuarray.to_gpu(surf_array[s].xiSort.astype(REAL))
        surf_array[s].yiDev      = gpuarray.to_gpu(surf_array[s].yiSort.astype(REAL))
        surf_array[s].ziDev      = gpuarray.to_gpu(surf_array[s].ziSort.astype(REAL))
        surf_array[s].xjDev      = gpuarray.to_gpu(surf_array[s].xjSort.astype(REAL))
        surf_array[s].yjDev      = gpuarray.to_gpu(surf_array[s].yjSort.astype(REAL))
        surf_array[s].zjDev      = gpuarray.to_gpu(surf_array[s].zjSort.astype(REAL))
        surf_array[s].AreaDev    = gpuarray.to_gpu(surf_array[s].AreaSort.astype(REAL))
        surf_array[s].vertexDev  = gpuarray.to_gpu(ravel(surf_array[s].vertex[surf_array[s].triangleSort]).astype(REAL))
        surf_array[s].xcDev      = gpuarray.to_gpu(ravel(surf_array[s].xcSort.astype(REAL)))
        surf_array[s].ycDev      = gpuarray.to_gpu(ravel(surf_array[s].ycSort.astype(REAL)))
        surf_array[s].zcDev      = gpuarray.to_gpu(ravel(surf_array[s].zcSort.astype(REAL)))
        surf_array[s].sizeTarDev = gpuarray.to_gpu(surf_array[s].sizeTarget.astype(int32))
        surf_array[s].offSrcDev  = gpuarray.to_gpu(surf_array[s].offsetSource.astype(int32))
        surf_array[s].offTwgDev  = gpuarray.to_gpu(ravel(surf_array[s].offsetTwigs.astype(int32)))
        surf_array[s].offMltDev  = gpuarray.to_gpu(ravel(surf_array[s].offsetMlt.astype(int32)))
        surf_array[s].M2P_lstDev = gpuarray.to_gpu(ravel(surf_array[s].M2P_list.astype(int32)))
        surf_array[s].P2P_lstDev = gpuarray.to_gpu(ravel(surf_array[s].P2P_list.astype(int32)))
        surf_array[s].xkDev      = gpuarray.to_gpu(surf_array[s].xk.astype(REAL))
        surf_array[s].wkDev      = gpuarray.to_gpu(surf_array[s].wk.astype(REAL))
        surf_array[s].kDev       = gpuarray.to_gpu((surf_array[s].sortSource%param.K).astype(int32))

    ind.indexDev = gpuarray.to_gpu(ind.index_large.astype(int32))

    Nfield = len(field_array)
    for f in range(Nfield):
        if len(field_array[f].xq)>0:
            field_array[f].xq_gpu = gpuarray.to_gpu(field_array[f].xq[:,0].astype(REAL))
            field_array[f].yq_gpu = gpuarray.to_gpu(field_array[f].xq[:,1].astype(REAL))
            field_array[f].zq_gpu = gpuarray.to_gpu(field_array[f].xq[:,2].astype(REAL))
            field_array[f].q_gpu  = gpuarray.to_gpu(field_array[f].q.astype(REAL))
