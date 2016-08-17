import numpy
import time

from pygbe.tree.FMMutils import addSources, sortPoints, generateTree, findTwigs
from pygbe.tree.direct import computeDiagonal
from pygbe.util.semi_analytical import GQ_1D
from pygbe.quadrature import quadratureRule_fine


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



class Field():
    """
    Field class.
    It contains the information about each region in the molecule.

    Attributes
    -----------

    parent: list, Pointer to "parent" surface.
    child : list, Pointer to "children" surfaces.
    LorY  : int, 1: Laplace, 2: Yukawa.
    kappa : float, inverse of Debye length.
    E     : float, dielectric constant.
    xq    : list, position of charges.
    q     : list, value of charges.
    coul  : int, 1: perform Coulomb interaction calculation, 0: don't do Coulomb.

    # Device data

    xq_gpu: list, x position of charges on GPU.
    yq_gpu: list, y position of charges on GPU.
    zq_gpu: list, z position of charges on GPU.
    q_gpu : list, value of charges on GPU.
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

    Attributes
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

    Attributes
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

    Attributes
    -----------

    II         : list, multipole order in the x-direction for the treecode.
    JJ         : list, multipole order in the y-direction for the treecode.
    KK         : list, multipole order in the z-direction for the treecode.
    index_large: list, pointers to the position of multipole order i, j, k
                       in the multipole array, organized in a 1D array of size
                       P*P*P. Index is given by index[i*P*P+j*P+k]
    index_small: list, pointers to the position of multipole order i, j, k
                       in the multipole array, organized in a 1D array which is
                       compressed with respect to index_large (does not consider
                       combinations of i,j,k which do not have a multipole).
    index      : list, copy of index_small
    index_ptr  : list, pointer to index_small. Data in index_small is organized
                      in a i-major fashion (i,j,k), and index_ptr points at the
                      position in index_small where the order i changes.
    combII     : array, combinatory of (I, i) where I is the maximum i multipole.
                       Used in coefficients of M2M.
    combJJ     : array, combinatory of (J, j) where J is the maximum j multipole.
                       Used in coefficients of M2M.
    combKK     : array, combinatory of (K, k) where K is the maximum k multipole.
                       Used in coefficients of M2M.
    IImii      : array, I-i where I is the maximum i multipole.
                       Used in exponents of M2M.
    JJmjj      : array, J-j where J is the maximum j multipole.
                       Used in exponents of M2M.
    KKmkk      : array, K-k where K is the maximum k multipole.
                       Used in exponents of M2M.

    # Device data

    indexDev   : list, index_large on GPU.
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
