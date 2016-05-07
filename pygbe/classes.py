import numpy
import time

class Event():
    """
    Class for logging like in pycuda's cuda.Event()
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
    It contains information about the solvent excluded surface.

    Attributes:
    -----------

    triangle      : list, indices to triangle vertices.
    vertex        : list, position of vertices.
    XinV          : list, weights input for single layer potential.
    XinK          : list, weights input for double layer potential.
    Xout_int      : list, output vector of interior operators.
    Xout_ext      : list, output vector of exterior operators.
    xi            : list, x component of center.
    yi            : list, y component of center.
    zi            : list, z component of center.
    xj            : list, x component of gauss nodes.
    yj            : list, y component of gauss nodes.
    zj            : list, z component of gauss nodes.
    Area          : list, areas of triangles.
    normal        : list, normal of triangles.
    sglInt_int    : list, singular integrals for V for internal equation.
    sglInt_ext    : list, singular integrals for V for external equation.
    xk            : list, position of gauss points on edges.
    wk            : list, weight of gauss points on edges.
    Xsk           : list, position of gauss points for near singular integrals.
    Wsk           : list, weight of gauss points for near singular integrals.
    tree          : list, tree structure.
    twig          : list, tree twigs.
    xiSort        : list, sorted x component of center.
    yiSort        : list, sorted y component of center.
    ziSort        : list, sorted z component of center.
    xjSort        : list, sorted x component of gauss nodes.
    yjSort        : list, sorted y component of gauss nodes.
    zjSort        : list, sorted z component of gauss nodes.
    xcSort        : list, sorted x component of the box centers according to
                          M2P_list array.
    ycSort        : list, sorted y component of the box centers according to
                          M2P_list array.
    zcSort        : list, sorted z component of the box centers according to
                          M2P_list array.
    AreaSort      : list, sorted array of areas.
    sglInt_intSort: list, sorted array of singular integrals for V for internal
                          equation.
    sglInt_extSort: list, sorted array of singular integrals for V for external
                          equation.
    unsort        : list, array of indices to unsort targets.
    triangleSort  : list, sorted array of triangles.
    sortTarget    : list, indices to sort targets.
    sortSource    : list, indices to sort sources.
    offsetSource  : list, offsets to sorted source array.
    offsetTarget  : list, offsets to sorted target array.
    sizeTarget    : list, number of targets per twig.
    offsetTwigs   : list, offset to twig in P2P list array.
    P2P_list      : list, pointers to twigs for P2P interaction list.
    offsetMlt     : list, offset to multipoles in M2P list array.
    M2P_list      : list, pointers to boxes for M2P interaction list.
    Precond       : list, sparse representation of preconditioner for self
                          interaction block.
    Ein           : float, permitivitty inside surface.
    Eout          : float, permitivitty outside surface.
    E_hat         : float, ratio of Ein/Eout.
    kappa_in      : float, kappa inside surface.
    kappa_out     : float, kappa inside surface.
    LorY_in       : int, Laplace (1) or Yukawa (2) in inner region.
    LorY_out      : int, Laplace (1) or Yukawa (2) in outer region.
    surf_type     : int, Surface type: internal_cavity (=0), stern or
                         dielecric_interface (=1).
    phi0          : list, known surface potential (dirichlet) or derivative of
                          potential (neumann).
    phi           : list, potential on surface.
    dphi          : list, derivative of potential on surface.

    # Device data:

    xiDev        : list, sorted x component of center (on the GPU).
    yiDev        : list, sorted y component of center (on the GPU).
    ziDev        : list, sorted z component of center (on the GPU).
    xjDev        : list, sorted x component of gauss nodes (on the GPU).
    yjDev        : list, sorted y component of gauss nodes (on the GPU).
    zjDev        : list, sorted z component of gauss nodes (on the GPU).
    xcDev        : list, sorted x component of the box centers according to
                         M2P_list array (on the GPU).
    ycDev        : list, sorted y component of the box centers according to
                         M2P_list array (on the GPU).
    zcDev        : list, sorted z component of the box centers according to
                         M2P_list array (on the GPU).
    AreaDev      : list, areas of triangles (on the GPU).
    sglInt_intDev: list, singular integrals for V for internal equation (on the 
                         GPU).
    sglInt_extDev: list, singular integrals for V for external equation (on the
                         GPU).
    vertexDev    : list, sorted vertex of the triangles.
    sizeTarDev   : list, number of targets per twig (on the GPU). 
    offSrcDev    : list, offsets to sorted source array (on the GPU).
    offMltDev    : list, offset to multipoles in M2P list array (on the GPU).
    offTwgDev    : list, offset to twig in P2P list array (on the GPU).
    M2P_lstDev   : list, pointers to boxes for M2P interaction list (on the GPU).
    P2P_lstDev   : list, pointers to twigs for P2P interaction list (on the GPU).
    xkDev        : list, position of gauss points on edges (on the GPU).
    wkDev        : list, weight of gauss points on edges (on the GPU).
    XskDev       : list, position of gauss points for near singular integrals 
                         (on the GPU).
    WskDev       : list, weight of gauss points for near singular integrals (on
                         the GPU).
    kDev         : list, quadrature number of each quadrature point, in order. 
                         (on the GPU)  
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
        self.sizeTarget   = []  # array with number of targets per twig
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
    Field class.
    It contains the information about each region in the molecule.

    Attributes:
    -----------

    parent: Pointer to "parent" surface.
    child : Pointer to "children" surfaces.
    LorY  : 1: Laplace, 2: Yukawa.
    kappa : inverse of Debye length.
    E     : dielectric constant.
    xq    : position of charges.
    q     : value of charges.
    coul  : 1: perform Coulomb interaction calculation, 0: don't do Coulomb.

    # Device data

    xq_gpu: x position of charges on GPU.
    yq_gpu: y position of charges on GPU.
    zq_gpu: z position of charges on GPU.
    q_gpu : value of charges on GPU.
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
    Timing class.
    It contains timing information for different parts of the code. 

    Attributes:
    -----------
    time_an   : float, time spent in compute the near singular integrals.  
    time_P2P  : float, time spent in compute the P2P part of the treecode.
    time_P2M  : float, time spent in compute the P2M part of the treecode.
    time_M2M  : float, time spent in compute the M2M part of the treecode.
    time_M2P  : float, time spent in compute the M2P part of the treecode.
    time_trans: float, time spent in transfer data to and from the GPU.
    time_sort : float, time spent in sorting data to send to the GPU.
    time_mass : float, time spent in compute the mass of the sources in treecode.
    AI_int    : int, counter of the amount of near singular integrals solved.
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
    Parameters class.
    It contains the information of the parameters needed to run the code. 
    
    Attributes:
    -----------

    kappa        :  float, inverse of Debye length.
    restart      :  int, Restart of GMRES.
    tol          :  float, Tolerance of GMRES.
    max_iter     :  int, Max number of GMRES iterations.
    P            :  int, Order of Taylor expansion.
    eps          :  int, Epsilon machine.
    Nm           :  int, Number of terms in Taylor expansion.
    NCRIT        :  int, Max number of targets per twig box.
    theta        :  float, MAC criterion for treecode.
    K            :  int, Number of Gauss points per element.
    K_fine       :  int, Number of Gauss points per element for near singular integrals.
    threshold    :  float, L/d criterion for semi-analytic intergrals.
    Nk           :  int, Gauss points per side for semi-analytical integrals.
    BSZ          :  int, CUDA block size.
    Nround       :  int, Max size of sorted target array.
    BlocksPerTwig:  int, Number of CUDA blocks that fit per tree twig.
    N            :  int, Total number of elements.
    Neq          :  int, Total number of equations.
    qe           :  float, Charge of an electron (1.60217646e-19).
    Na           :  float, Avogadro's number (6.0221415e23).
    E_0          :  float, Vacuum dielectric constant (8.854187818e-12).
    REAL         :  Data type.
    E_field      :  list, Regions where energy will be calculated.
    GPU          :  int, =1: with GPU, =0: no GPU.
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
    """
    It contains the precompute indices required for the treecode computation.
    
    Attributes:
    -----------

    II         : list, multipole order in the x-direction for the treecode.
    JJ         : list, multipole order in the y-direction for the treecode. 
    KK         : list, multipole order in the z-direction for the treecode. 
    index      : list, pointers to the location of the mulipole of order i,j,k 
                       in the multipole array. 
    index_small:
    index_large:
    index_ptr  :
    combII     :
    combJJ     : 
    combKK     : 
    IImii      :   
    JJmjj      : 
    KKmkk      : 

    # Device data

    indexDev   :
    """
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
