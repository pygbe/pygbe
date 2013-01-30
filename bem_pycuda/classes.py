from numpy import *
import sys
sys.path.append('tree')
from FMMutils import *
from direct   import computeDiagonal
sys.path.append('../util')
from semi_analytical    import *
from triangulation      import *
from readData           import readVertex, readTriangle, readpqr, readcrd, readFields, readSurf

# PyCUDA libraries
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

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
        self.sglIntY  = []  # singular integrals for V for Yukawa
        self.sglIntL  = []  # singular integrals for V for Laplace
        self.xk       = []  # position of gauss points on edges
        self.wk       = []  # weight of gauss points on edges
        self.Xsk      = []  # position of gauss points for near singular integrals
        self.Wsk      = []  # weight of gauss points for near singular integrals
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
        self.sglIntYSort  = []  # sorted array of singular integrals for V for Yukawa
        self.sglIntLSort  = []  # sorted array of singular integrals for V for Laplace
        self.unsort       = []  # array of indices to unsort targets
        self.triangleSort = []  # sorted array of triangles
        self.sortTarget   = []  # array of indices to sort targets
        self.sortSource   = []  # array of indices to sort sources
        self.offsetSource = []  # array with offsets to sorted source array
        self.offsetTarget = []  # array with offsets to sorted target array
        self.sizeTarget   = []  # array with number of targets pero twig
        self.offsetTwigs  = []  # offset to twig in P2P list array
        self.P2P_list     = []  # pointers to twigs for P2P interaction list
        self.offsetMlt    = []  # offset to multipoles in M2P list array
        self.M2P_list     = []  # pointers to boxes for M2P interaction list
        self.Precond      = []  # Sparse representation of preconditioner for self interaction block
        self.E_hat        = 0   # ratio of Ein/Eout
        self.kappa_in     = 0   # kappa inside surface
        self.kappa_out    = 0   # kappa inside surface

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
        self.sglIntYDev = []
        self.sglIntLDev = []
        self.vertexDev  = []
        self.sizeTarDev = []
        self.offSrcDev  = []
        self.offMltDev  = []
        self.offTwgDev  = []
        self.M2P_lstDev = []
        self.P2P_lstDev = []
        self.xkDev      = []
        self.wkDev      = []
        self.XskDev     = []
        self.WskDev     = []
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
        self.K_fine        = 0               # Number of Gauss points per element for near singular integrals
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
        xi[:,0] = average(y[triangle[:],:,0], axis=1)
        xi[:,1] = average(y[triangle[:],:,1], axis=1)
        xi[:,2] = average(y[triangle[:],:,2], axis=1)

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

def quadratureRule_fine(K):
    
    # 1 Gauss point
    if K==1:
        X = array([1/3., 1/3., 1/3.])
        W = array([1])

    # 7 Gauss points
    if K==7:
        a = 1/3.
        b1 = 0.059715871789770; b2 = 0.470142064105115
        c1 = 0.797426985353087; c2 = 0.101286507323456

        wa = 0.225000000000000
        wb = 0.132394152788506
        wc = 0.125939180544827

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1])
        W = array([wa,wb,wb,wb,wc,wc,wc])
        
   
    # 13 Gauss points
    if K==13:
        a = 1/3.
        b1 = 0.479308067841920; b2 = 0.260345966079040
        c1 = 0.869739794195568; c2 = 0.065130102902216
        d1 = 0.048690315425316; d2 = 0.312865496004874; d3 = 0.638444188569810
        wa = -0.149570044467682
        wb = 0.175615257433208
        wc = 0.053347235608838
        wd = 0.077113760890257

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1, 
                    c1,c2,c2,c2,c1,c2,c2,c2,c1, 
                    d1,d2,d3,d1,d3,d2,d2,d1,d3,d2,d3,d1,d3,d1,d2,d3,d2,d1])
        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,wd,wd,wd])
        
    # 17 Gauss points
    if K==17:
        a = 1/3.
        b1 = 0.081414823414554; b2 = 0.459292588292723
        c1 = 0.658861384496480; c2 = 0.170569307751760
        d1 = 0.898905543365938; d2 = 0.050547228317031
        e1 = 0.008394777409958; e2 = 0.263112829634638; e3 = 0.728492392955404
        wa = 0.144315607677787
        wb = 0.095091634267285
        wc = 0.103217370534718
        wd = 0.032458497623198
        we = 0.027230314174435

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1, 
                    c1,c2,c2,c2,c1,c2,c2,c2,c1, 
                    d1,d2,d2,d2,d1,d2,d2,d2,d1, 
                    e1,e2,e3,e1,e3,e2,e2,e1,e3,e2,e3,e1,e3,e1,e2,e3,e2,e1])
                    
        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,
                    we,we,we,we,we,we])

    # 19 Gauss points
    if K==19:
        a = 1/3.
        b1 = 0.020634961602525; b2 = 0.489682519198738
        c1 = 0.125820817014127; c2 = 0.437089591492937
        d1 = 0.623592928761935; d2 = 0.188203535619033
        e1 = 0.910540973211095; e2 = 0.044729513394453
        f1 = 0.036838412054736; f2 = 0.221962989160766; f3 = 0.741198598784498

        wa = 0.097135796282799
        wb = 0.031334700227139
        wc = 0.077827541004774
        wd = 0.079647738927210
        we = 0.025577675658698
        wf = 0.043283539377289

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1,
                    d1,d2,d2,d2,d1,d2,d2,d2,d1,
                    e1,e2,e2,e2,e1,e2,e2,e2,e1,
                    f1,f2,f3,f1,f3,f2,f2,f1,f3,f2,f3,f1,f3,f1,f2,f3,f2,f1])
        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,
                    we,we,we,
                    wf,wf,wf,wf,wf,wf])

    # 79 Gauss points
    if K==79:
        a  = 1/3.
        b1 = -0.001900928704400; b2 = 0.500950464352200
        c1 = 0.023574084130543; c2 = 0.488212957934729
        d1 = 0.089726636099435; d2 = 0.455136681950283
        e1 = 0.196007481363421; e2 = 0.401996259318289
        f1 = 0.488214180481157; f2 = 0.255892909759421
        g1 = 0.647023488009788; g2 = 0.176488255995106
        h1 = 0.791658289326483; h2 = 0.104170855336758
        i1 = 0.893862072318140; i2 = 0.053068963840930
        j1 = 0.916762569607942; j2 = 0.041618715196029
        k1 = 0.976836157186356; k2 = 0.011581921406822
        l1 = 0.048741583664839; l2 = 0.344855770229001; l3 = 0.606402646106160
        m1 = 0.006314115948605; m2 = 0.377843269594854; m3 = 0.615842614456541
        n1 = 0.134316520547348; n2 = 0.306635479062357; n3 = 0.559048000390295
        o1 = 0.013973893962392; o2 = 0.249419362774742; o3 = 0.736606743262866
        p1 = 0.075549132909764; p2 = 0.212775724802802; p3 = 0.711675142287434
        q1 = -0.008368153208227; q2 = 0.146965436053239; q3 = 0.861402717154987
        r1 = 0.026686063258714; r2 = 0.137726978828923; r3 = 0.835586957912363
        s1 = 0.010547719294141; s2 = 0.059696109149007; s3 = 0.929756171556853

        wa = 0.033057055541624
        wb = 0.000867019185663
        wc = 0.011660052716448
        wd = 0.022876936356421
        we = 0.030448982673938
        wf = 0.030624891725355
        wg = 0.024368057676800
        wh = 0.015997432032024
        wi = 0.007698301815602
        wj = -0.000632060497488
        wk = 0.001751134301193
        wl = 0.016465839189576
        wm = 0.004839033540485
        wn = 0.025804906534650
        wo = 0.008471091054441
        wp = 0.018354914106280
        wq = 0.000704404677908
        wr = 0.010112684927462
        ws = 0.003573909385950

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1,
                    d1,d2,d2,d2,d1,d2,d2,d2,d1,
                    e1,e2,e2,e2,e1,e2,e2,e2,e1,
                    f1,f2,f2,f2,f1,f2,f2,f2,f1,
                    g1,g2,g2,g2,g1,g2,g2,g2,g1,
                    h1,h2,h2,h2,h1,h2,h2,h2,h1,
                    i1,i2,i2,i2,i1,i2,i2,i2,i1,
                    j1,j2,j2,j2,j1,j2,j2,j2,j1,
                    k1,k2,k2,k2,k1,k2,k2,k2,k1,
                    l1,l2,l3,l1,l3,l2,l2,l1,l3,l2,l3,l1,l3,l1,l2,l3,l2,l1,
                    m1,m2,m3,m1,m3,m2,m2,m1,m3,m2,m3,m1,m3,m1,m2,m3,m2,m1,
                    n1,n2,n3,n1,n3,n2,n2,n1,n3,n2,n3,n1,n3,n1,n2,n3,n2,n1,
                    o1,o2,o3,o1,o3,o2,o2,o1,o3,o2,o3,o1,o3,o1,o2,o3,o2,o1,
                    p1,p2,p3,p1,p3,p2,p2,p1,p3,p2,p3,p1,p3,p1,p2,p3,p2,p1,
                    q1,q2,q3,q1,q3,q2,q2,q1,q3,q2,q3,q1,q3,q1,q2,q3,q2,q1,
                    r1,r2,r3,r1,r3,r2,r2,r1,r3,r2,r3,r1,r3,r1,r2,r3,r2,r1,
                    s1,s2,s3,s1,s3,s2,s2,s1,s3,s2,s3,s1,s3,s1,s2,s3,s2,s1])

        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,
                    we,we,we,
                    wf,wf,wf,
                    wg,wg,wg,
                    wh,wh,wh,
                    wi,wi,wi,
                    wj,wj,wj,
                    wk,wk,wk,
                    wl,wl,wl,wl,wl,wl,
                    wm,wm,wm,wm,wm,wm,
                    wn,wn,wn,wn,wn,wn,
                    wo,wo,wo,wo,wo,wo,
                    wp,wp,wp,wp,wp,wp,
                    wq,wq,wq,wq,wq,wq,
                    wr,wr,wr,wr,wr,wr,
                    ws,ws,ws,ws,ws,ws])

    return X, W

def fill_surface(surf,param):

    N  = len(surf.triangle)
    Nj = N*param.K
    # Calculate centers
    surf.xi = average(surf.vertex[surf.triangle[:],:,0], axis=1)
    surf.yi = average(surf.vertex[surf.triangle[:],:,1], axis=1)
    surf.zi = average(surf.vertex[surf.triangle[:],:,2], axis=1)

    surf.normal = zeros((N,3))
    surf.Area = zeros(N)

    L0 = surf.vertex[surf.triangle[:,1]] - surf.vertex[surf.triangle[:,0]]
    L2 = surf.vertex[surf.triangle[:,0]] - surf.vertex[surf.triangle[:,2]]
    surf.normal = cross(L0,L2)
    surf.Area = sqrt(surf.normal[:,0]**2 + surf.normal[:,1]**2 + surf.normal[:,2]**2)/2
    surf.normal[:,0] = surf.normal[:,0]/(2*surf.Area)
    surf.normal[:,1] = surf.normal[:,1]/(2*surf.Area)
    surf.normal[:,2] = surf.normal[:,2]/(2*surf.Area)

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

    surf.xk,surf.wk = GQ_1D(param.Nk)
    surf.Xsk,surf.Wsk = quadratureRule_fine(param.K_fine) 


    # Generate preconditioner
    # Will use block-diagonal preconditioner (AltmanBardhanWhiteTidor2008)
    surf.Precond = zeros((4,N))  # Stores the inverse of the block diagonal (also a tridiag matrix)
                                 # Order: Top left, top right, bott left, bott right    
    VL = zeros(N) # Top left block
    KL = zeros(N) # Top right block
    VY = zeros(N) # Bottom left block
    KY = zeros(N) # Bottom right block
    centers = zeros((N,3))
    centers[:,0] = surf.xi[:]
    centers[:,1] = surf.yi[:]
    centers[:,2] = surf.zi[:]
    computeDiagonal(VL, KL, VY, KY, ravel(surf.vertex[surf.triangle[:]]), ravel(centers), 
                    surf.kappa_out, 2*pi, 0., surf.xk, surf.wk)

    dX11 = KL
    dX12 = -VL
    dX21 = KY
    dX22 = surf.E_hat*VY
    surf.sglIntL = VL # Array for singular integral of V for Laplace
    surf.sglIntY = VY # Array for singular integral of V for Yukawa

    d_aux = 1/(dX22-dX21*dX12/dX11)
    surf.Precond[0,:] = 1/dX11 + 1/dX11*dX12*d_aux*dX21/dX11
    surf.Precond[1,:] = -1/dX11*dX12*d_aux
    surf.Precond[2,:] = -d_aux*dX21/dX11
    surf.Precond[3,:] = d_aux

    tic = time.time()
    sortPoints(surf, surf.tree, surf.twig, param)
    toc = time.time()
    time_sort = toc - tic

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
        tic = time.time()
        s.vertex = readVertex(files[i]+'.vert', param.REAL)
        triangle_raw = readTriangle(files[i]+'.face')
        toc = time.time()
        print 'Time load mesh: %f'%(toc-tic)
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

def dataTransfer(surf_array, field_array, ind, param, kernel):

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
        surf_array[s].sglIntLDev = gpuarray.to_gpu(surf_array[s].sglIntLSort.astype(REAL))
        surf_array[s].sglIntYDev = gpuarray.to_gpu(surf_array[s].sglIntYSort.astype(REAL))
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
        surf_array[s].XskDev     = gpuarray.to_gpu(surf_array[s].Xsk.astype(REAL))
        surf_array[s].WskDev     = gpuarray.to_gpu(surf_array[s].Wsk.astype(REAL))
        surf_array[s].kDev       = gpuarray.to_gpu((surf_array[s].sortSource%param.K).astype(int32))

    ind.indexDev = gpuarray.to_gpu(ind.index_large.astype(int32))

    Nfield = len(field_array)
    for f in range(Nfield):
        if len(field_array[f].xq)>0:
            field_array[f].xq_gpu = gpuarray.to_gpu(field_array[f].xq[:,0].astype(REAL))
            field_array[f].yq_gpu = gpuarray.to_gpu(field_array[f].xq[:,1].astype(REAL))
            field_array[f].zq_gpu = gpuarray.to_gpu(field_array[f].xq[:,2].astype(REAL))
            field_array[f].q_gpu  = gpuarray.to_gpu(field_array[f].q.astype(REAL))


