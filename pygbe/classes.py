import numpy
import time

class Event():
    """
    class for logging like in pycuda's cuda.Event()
    """
    def __init__(self):
        self.t = 0
    def record(self):
        t = time.time()*1e3
    def time_till(self,toc):
        return toc.t-self.t
    def synchronize(self):
        pass


class Surface():
    """
    Surface class.

    Attributes:
    -----------

    triangle      : indices to triangle vertices
    vertex        : position of vertices
    XinV          : weights input for single layer potential
    XinK          : weights input for double layer potential
    Xout_int      : output vector of interior operators
    Xout_ext      : output vector of exterior operators
    xi            : x component of center
    yi            : y component of center
    zi            : z component of center
    xj            : x component of gauss nodes
    yj            : y component of gauss nodes
    zj            : z component of gauss nodes
    Area          : Area of triangles
    normal        : normal of triangles
    sglInt_int    : singular integrals for V for internal equation
    sglInt_ext    : singular integrals for V for external equation
    xk            : position of gauss points on edges
    wk            : weight of gauss points on edges
    Xsk           : position of gauss points for near singular integrals
    Wsk           : weight of gauss points for near singular integrals
    tree          : tree structure
    twig          : tree twigs
    xiSort        : sorted x component of center
    yiSort        : sorted y component of center
    ziSort        : sorted z component of center
    xjSort        : sorted x component of gauss nodes
    yjSort        : sorted y component of gauss nodes
    zjSort        : sorted z component of gauss nodes
    xcSort        : sorted x component box centers according to M2P_list array
    ycSort        : sorted y component box centers according to M2P_list array
    zcSort        : sorted z component box centers according to M2P_list array
    AreaSort      : sorted array of areas
    sglInt_intSort: sorted array of singular integrals for V for internal equation
    sglInt_extSort: sorted array of singular integrals for V for external equation
    unsort        : array of indices to unsort targets
    triangleSort  : sorted array of triangles
    sortTarget    : array of indices to sort targets
    sortSource    : array of indices to sort sources
    offsetSource  : array with offsets to sorted source array
    offsetTarget  : array with offsets to sorted target array
    sizeTarget    : array with number of targets pero twig
    offsetTwigs   : offset to twig in P2P list array
    P2P_list      : pointers to twigs for P2P interaction list
    offsetMlt     : offset to multipoles in M2P list array
    M2P_list      : pointers to boxes for M2P interaction list
    Precond       : Sparse representation of preconditioner for self interaction block
    Ein           : Permitivitty inside surface
    Eout          : Permitivitty outside surface
    E_hat         : ratio of Ein/Eout
    kappa_in      : kappa inside surface
    kappa_out     : kappa inside surface
    LorY_in       : Laplace or Yukawa in inner region
    LorY_out      : Laplace or Yukawa in outer region
    surf_type     : Surface type: internal_cavity (=0), stern or dielecric_interface (=1)
    phi0          : Known surface potential (dirichlet) or derivative of potential (neumann)
    phi           : Potential on surface
    dphi          : Derivative of potential on surface

    # Device data

    xiDev        :
    yiDev        :
    ziDev        :
    xjDev        :
    yjDev        :
    zjDev        :
    xcDev        :
    ycDev        :
    zcDev        :
    AreaDev      :
    sglInt_intDev:
    sglInt_extDev:
    vertexDev    :
    sizeTarDev   :
    offSrcDev    :
    offMltDev    :
    offTwgDev    :
    M2P_lstDev   :
    P2P_lstDev   :
    xkDev        :
    wkDev        :
    XskDev       :
    WskDev       :
    kDev         :

    """
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
        self.sglInt_int = []  # singular integrals for V for internal equation
        self.sglInt_ext = []  # singular integrals for V for external equation
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
        self.sglInt_intSort  = []  # sorted array of singular integrals for V for internal equation
        self.sglInt_extSort  = []  # sorted array of singular integrals for V for external equation
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
        self.Ein          = 0   # Permitivitty inside surface
        self.Eout         = 0   # Permitivitty outside surface
        self.E_hat        = 0   # ratio of Ein/Eout
        self.kappa_in     = 0   # kappa inside surface
        self.kappa_out    = 0   # kappa inside surface
        self.LorY_in      = 0   # Laplace or Yukawa in inner region
        self.LorY_out     = 0   # Laplace or Yukawa in outer region
        self.surf_type    = 0   # Surface type: internal_cavity (=0), stern or dielecric_interface (=1)
        self.phi0         = []  # Known surface potential (dirichlet) or derivative of potential (neumann)
        self.phi          = []  # Potential on surface
        self.dphi         = []  # Derivative of potential on surface

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
        self.sglInt_intDev = []
        self.sglInt_extDev = []
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


class Field():
    """
    Class for 

    Attributes:
    -----------

    parent: Pointer to "parent" surface
    child : Pointer to "children" surfaces
    LorY  : 1: Laplace, 2: Yukawa
    kappa : inverse of Debye length
    E     : dielectric constant
    xq    : position of charges
    q     : value of charges
    coul  : 1: perform Coulomb interaction calculation, 0: don't do Coulomb.

    # Device data

    xq_gpu: x position of charges on gpu
    yq_gpu: y position of charges on gpu
    zq_gpu: z position of charges on gpu
    q_gpu : value of charges on gpu

    """
    def __init__(self):
        self.parent = []    # Pointer to "parent" surface
        self.child  = []    # Pointer to "children" surfaces
        self.LorY   = []    # 1: Laplace, 2: Yukawa
        self.kappa  = []    # inverse of Debye length
        self.E      = []    # dielectric constant
        self.xq     = []    # position of charges
        self.q      = []    # value of charges
        self.coul   = []    # 1: perform Coulomb interaction calculation
                            # 0: don't do Coulomb calculation

        # Device data
        self.xq_gpu = []    # x position of charges on gpu
        self.yq_gpu = []    # y position of charges on gpu
        self.zq_gpu = []    # z position of charges on gpu
        self.q_gpu  = []    # value of charges on gpu


class Timing():
    """
    Class for 

    Attributes:
    -----------
    time_an   : 
    time_P2P  : 
    time_P2M  : 
    time_M2M  : 
    time_M2P  : 
    time_trans: 
    time_sort : 
    time_mass : 
    AI_int    : 

    """
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


class Parameters():
    """
    Class for 
    """
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
        self.Neq           = 0               # Total number of equations
        self.qe            = 1.60217646e-19  # Charge of an electron
        self.Na            = 6.0221415e23    # Avogadro's number
        self.E_0           = 8.854187818e-12 # Vacuum dielectric constant
        self.REAL          = 0               # Data type
        self.E_field       = []              # Regions where energy will be calculated
        self.GPU           = -1              # =1: with GPU, =0: no GPU


class IndexConstant():
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
