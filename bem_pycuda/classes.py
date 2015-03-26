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

class fields():
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
        self.Neq           = 0               # Total number of equations
        self.qe            = 1.60217646e-19  # Charge of an electron
        self.Na            = 6.0221415e23    # Avogadro's number
        self.E_0           = 8.854187818e-12 # Vacuum dielectric constant
        self.REAL          = 0               # Data type
        self.E_field       = []              # Regions where energy will be calculated
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

def computePrecond(surf):

    # Generate preconditioner
    # Will use block-diagonal preconditioner (AltmanBardhanWhiteTidor2008)
    N = len(surf.triangle)
    surf.Precond = zeros((4,N))  # Stores the inverse of the block diagonal (also a tridiag matrix)
                                 # Order: Top left, top right, bott left, bott right    
    centers = zeros((N,3))
    centers[:,0] = surf.xi[:]
    centers[:,1] = surf.yi[:]
    centers[:,2] = surf.zi[:]

#   Compute diagonal integral for internal equation
    VL = zeros(N) 
    KL = zeros(N) 
    VY = zeros(N)
    KY = zeros(N)
    computeDiagonal(VL, KL, VY, KY, ravel(surf.vertex[surf.triangle[:]]), ravel(centers), 
                    surf.kappa_in, 2*pi, 0., surf.xk, surf.wk)
    if surf.LorY_in == 1:
        dX11 = KL
        dX12 = -VL
        surf.sglInt_int = VL # Array for singular integral of V through interior
    elif surf.LorY_in == 2:
        dX11 = KY
        dX12 = -VY
        surf.sglInt_int = VY # Array for singular integral of V through interior
    else:
        surf.sglInt_int = zeros(N)

#   Compute diagonal integral for external equation
    VL = zeros(N) 
    KL = zeros(N) 
    VY = zeros(N)
    KY = zeros(N)
    computeDiagonal(VL, KL, VY, KY, ravel(surf.vertex[surf.triangle[:]]), ravel(centers), 
                    surf.kappa_out, 2*pi, 0., surf.xk, surf.wk)
    if surf.LorY_out == 1:
        dX21 = KL
        dX22 = surf.E_hat*VL
        surf.sglInt_ext = VL # Array for singular integral of V through exterior
    elif surf.LorY_out == 2:
        dX21 = KY
        dX22 = surf.E_hat*VY
        surf.sglInt_ext = VY # Array for singular integral of V through exterior
    else:
        surf.sglInt_ext = zeros(N)

    if surf.surf_type!='dirichlet_surface' and surf.surf_type!='neumann_surface':
        d_aux = 1/(dX22-dX21*dX12/dX11)
        surf.Precond[0,:] = 1/dX11 + 1/dX11*dX12*d_aux*dX21/dX11
        surf.Precond[1,:] = -1/dX11*dX12*d_aux
        surf.Precond[2,:] = -d_aux*dX21/dX11
        surf.Precond[3,:] = d_aux
    elif surf.surf_type=='dirichlet_surface':
        surf.Precond[0,:] = 1/VY  # So far only for Yukawa outside
    elif surf.surf_type=='neumann_surface' or surf.surf_type=='asc_surface':
        surf.Precond[0,:] = 1/(2*pi)


def getGaussPoints(y,triangle, n):
    # y         : vertices
    # triangle  : array with indices for corresponding triangles
    # n         : Gauss points per element

    N  = len(triangle) # Number of triangles
    xi = zeros((N*n,3))
    if n==1:
        xi[:,0] = average(y[triangle[:],0], axis=1)
        xi[:,1] = average(y[triangle[:],1], axis=1)
        xi[:,2] = average(y[triangle[:],2], axis=1)

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
            xi[n*i+1,:] = dot(M, array([.797426985353087,.101286507323456,.101286507323456]))
            xi[n*i+2,:] = dot(M, array([.101286507323456,.797426985353087,.101286507323456]))
            xi[n*i+3,:] = dot(M, array([.101286507323456,.101286507323456,.797426985353087]))
            xi[n*i+4,:] = dot(M, array([.059715871789770,.470142064105115,.470142064105115]))
            xi[n*i+5,:] = dot(M, array([.470142064105115,.059715871789770,.470142064105115]))
            xi[n*i+6,:] = dot(M, array([.470142064105115,.470142064105115,.059715871789770]))

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

    # 25 Gauss points
    if K==25:
        a  = 1/3.
        b1 = 0.028844733232685; b2 = 0.485577633383657
        c1 = 0.781036849029926; c2 = 0.109481575485037
        d1 = 0.141707219414880; d2 = 0.307939838764121; d3 = 0.550352941820999
        e1 = 0.025003534762686; e2 = 0.246672560639903; e3 = 0.728323904597411
        f1 = 0.009540815400299; f2 = 0.066803251012200; f3 = 0.923655933587500
        
        wa = 0.090817990382754
        wb = 0.036725957756467
        wc = 0.045321059435528
        wd = 0.072757916845420
        we = 0.028327242531057
        wf = 0.009421666963733

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1,
                    d1,d2,d3,d1,d3,d2,d2,d1,d3,d2,d3,d1,d3,d1,d2,d3,d2,d1,
                    e1,e2,e3,e1,e3,e2,e2,e1,e3,e2,e3,e1,e3,e1,e2,e3,e2,e1,
                    f1,f2,f3,f1,f3,f2,f2,f1,f3,f2,f3,f1,f3,f1,f2,f3,f2,f1])
        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,wd,wd,wd,
                    we,we,we,we,we,we,
                    wf,wf,wf,wf,wf,wf])

    # 37 Gauss points
    if K==37:
        a = 1/3.
        b1 = 0.009903630120591; b2 = 0.495048184939705
        c1 = 0.062566729780852; c2 = 0.468716635109574
        d1 = 0.170957326397447; d2 = 0.414521336801277
        e1 = 0.541200855914337; e2 = 0.229399572042831
        f1 = 0.771151009607340; f2 = 0.114424495196330
        g1 = 0.950377217273082; g2 = 0.024811391363459
        h1 = 0.094853828379579; h2 = 0.268794997058761; h3 = 0.636351174561660
        i1 = 0.018100773278807; i2 = 0.291730066734288; i3 = 0.690169159986905
        j1 = 0.022233076674090; j2 = 0.126357385491669; j3 = 0.851409537834241

        wa = 0.052520923400802
        wb = 0.011280145209330 
        wc = 0.031423518362454
        wd = 0.047072502504194 
        we = 0.047363586536355
        wf = 0.031167529045794 
        wg = 0.007975771465074
        wh = 0.036848402728732 
        wi = 0.017401463303822
        wj = 0.015521786839045

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1,
                    d1,d2,d2,d2,d1,d2,d2,d2,d1,
                    e1,e2,e2,e2,e1,e2,e2,e2,e1,
                    f1,f2,f2,f2,f1,f2,f2,f2,f1,
                    g1,g2,g2,g2,g1,g2,g2,g2,g1,
                    h1,h2,h3,h1,h3,h2,h2,h1,h3,h2,h3,h1,h3,h1,h2,h3,h2,h1,
                    i1,i2,i3,i1,i3,i2,i2,i1,i3,i2,i3,i1,i3,i1,i2,i3,i2,i1,
                    j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1])

        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,
                    we,we,we,
                    wf,wf,wf,
                    wg,wg,wg,
                    wh,wh,wh,wh,wh,wh,
                    wi,wi,wi,wi,wi,wi,
                    wj,wj,wj,wj,wj,wj])

    # 48 Gauss points
    if K==48:
        a1 =-0.013945833716486; a2 = 0.506972916858243
        b1 = 0.137187291433955; b2 = 0.431406354283023
        c1 = 0.444612710305711; c2 = 0.277693644847144
        d1 = 0.747070217917492; d2 = 0.126464891041254
        e1 = 0.858383228050628; e2 = 0.070808385974686
        f1 = 0.962069659517853; f2 = 0.018965170241073
        g1 = 0.133734161966621; g2 = 0.261311371140087; g3 = 0.604954466893291
        h1 = 0.036366677396917; h2 = 0.388046767090269; h3 = 0.575586555512814
        i1 =-0.010174883126571; i2 = 0.285712220049916; i3 = 0.724462663076655
        j1 = 0.036843869875878; j2 = 0.215599664072284; j3 = 0.747556466051838
        k1 = 0.012459809331199; k2 = 0.103575616576386; k3 = 0.883964574092416        

        wa = 0.001916875642849
        wb = 0.044249027271145 
        wc = 0.051186548718852
        wd = 0.023687735870688 
        we = 0.013289775690021
        wf = 0.004748916608192 
        wg = 0.038550072599593
        wh = 0.027215814320624 
        wi = 0.002182077366797
        wj = 0.021505319847731
        wk = 0.007673942631049

        X = array([a1,a2,a2,a2,a1,a2,a2,a2,a1,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1,
                    d1,d2,d2,d2,d1,d2,d2,d2,d1,
                    e1,e2,e2,e2,e1,e2,e2,e2,e1,
                    f1,f2,f2,f2,f1,f2,f2,f2,f1,
                    g1,g2,g3,g1,g3,g2,g2,g1,g3,g2,g3,g1,g3,g1,g2,g3,g2,g1,
                    h1,h2,h3,h1,h3,h2,h2,h1,h3,h2,h3,h1,h3,h1,h2,h3,h2,h1,
                    i1,i2,i3,i1,i3,i2,i2,i1,i3,i2,i3,i1,i3,i1,i2,i3,i2,i1,
                    j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1,
                    k1,k2,k3,k1,k3,k2,k2,k1,k3,k2,k3,k1,k3,k1,k2,k3,k2,k1])

        W = array([wa,wa,wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,
                    we,we,we,
                    wf,wf,wf,
                    wg,wg,wg,wg,wg,wg,
                    wh,wh,wh,wh,wh,wh,
                    wi,wi,wi,wi,wi,wi,
                    wj,wj,wj,wj,wj,wj,
                    wk,wk,wk,wk,wk,wk])


    # 52 Gauss points
    if K==52:
        a = 1/3.
        b1 = 0.005238916103123; b2 = 0.497380541948438
        c1 = 0.173061122901295; c2 = 0.413469438549352
        d1 = 0.059082801866017; d2 = 0.470458599066991
        e1 = 0.518892500060958; e2 = 0.240553749969521
        f1 = 0.704068411554854; f2 = 0.147965794222573
        g1 = 0.849069624685052; g2 = 0.075465187657474
        h1 = 0.966807194753950; h2 = 0.016596402623025
        i1 = 0.103575692245252; i2 = 0.296555596579887; i3 = 0.599868711174861
        j1 = 0.020083411655416; j2 = 0.337723063403079; j3 = 0.642193524941505
        k1 =-0.004341002614139; k2 = 0.204748281642812; k3 = 0.799592720971327
        l1 = 0.041941786468010; l2 = 0.189358492130623; l3 = 0.768699721401368
        m1 = 0.014317320230681; m2 = 0.085283615682657; m3 = 0.900399064086661 

        wa = 0.046875697427642 
        wb = 0.006405878578585 
        wc = 0.041710296739387
        wd = 0.026891484250064 
        we = 0.042132522761650
        wf = 0.030000266842773 
        wg = 0.014200098925024
        wh = 0.003582462351273 
        wi = 0.032773147460627
        wj = 0.015298306248441 
        wk = 0.002386244192839
        wl = 0.019084792755899 
        wm = 0.006850054546542

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1,
                    d1,d2,d2,d2,d1,d2,d2,d2,d1,
                    e1,e2,e2,e2,e1,e2,e2,e2,e1,
                    f1,f2,f2,f2,f1,f2,f2,f2,f1,
                    g1,g2,g2,g2,g1,g2,g2,g2,g1,
                    h1,h2,h2,h2,h1,h2,h2,h2,h1,
                    i1,i2,i3,i1,i3,i2,i2,i1,i3,i2,i3,i1,i3,i1,i2,i3,i2,i1,
                    j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1,
                    k1,k2,k3,k1,k3,k2,k2,k1,k3,k2,k3,k1,k3,k1,k2,k3,k2,k1,
                    l1,l2,l3,l1,l3,l2,l2,l1,l3,l2,l3,l1,l3,l1,l2,l3,l2,l1,
                    m1,m2,m3,m1,m3,m2,m2,m1,m3,m2,m3,m1,m3,m1,m2,m3,m2,m1])

        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,
                    we,we,we,
                    wf,wf,wf,
                    wg,wg,wg,
                    wh,wh,wh,
                    wi,wi,wi,wi,wi,wi,
                    wj,wj,wj,wj,wj,wj,
                    wk,wk,wk,wk,wk,wk,
                    wl,wl,wl,wl,wl,wl,
                    wm,wm,wm,wm,wm,wm])

    # 61 Gauss points
    if K==61:
        a = 1/3.
        b1 = 0.005658918886452; b2 = 0.497170540556774
        c1 = 0.035647354750751; c2 = 0.482176322624625
        d1 = 0.099520061958437; d2 = 0.450239969020782
        e1 = 0.199467521245206; e2 = 0.400266239377397
        f1 = 0.495717464058095; f2 = 0.252141267970953
        g1 = 0.675905990683077; g2 = 0.162047004658461
        h1 = 0.848248235478508; h2 = 0.075875882260746
        i1 = 0.968690546064356; i2 = 0.015654726967822
        j1 = 0.010186928826919; j2 = 0.334319867363658; j3 = 0.655493203809423
        k1 = 0.135440871671036; k2 = 0.292221537796944; k3 = 0.572337590532020
        l1 = 0.054423924290583; l2 = 0.319574885423190; l3 = 0.626001190286228
        m1 = 0.012868560833637; m2 = 0.190704224192292; m3 = 0.796427214974071
        n1 = 0.067165782413524; n2 = 0.180483211648746; n3 = 0.752351005937729
        o1 = 0.014663182224828; o2 = 0.080711313679564; o3 = 0.904625504095608

        wa = 0.033437199290803
        wb = 0.005093415440507 
        wc = 0.014670864527638
        wd = 0.024350878353672 
        we = 0.031107550868969
        wf = 0.031257111218620 
        wg = 0.024815654339665
        wh = 0.014056073070557 
        wi = 0.003194676173779
        wj = 0.008119655318993 
        wk = 0.026805742283163
        wl = 0.018459993210822 
        wm = 0.008476868534328
        wn = 0.018292796770025
        wo = 0.006665632004165

        X = array([a,a,a,
                    b1,b2,b2,b2,b1,b2,b2,b2,b1,
                    c1,c2,c2,c2,c1,c2,c2,c2,c1,
                    d1,d2,d2,d2,d1,d2,d2,d2,d1,
                    e1,e2,e2,e2,e1,e2,e2,e2,e1,
                    f1,f2,f2,f2,f1,f2,f2,f2,f1,
                    g1,g2,g2,g2,g1,g2,g2,g2,g1,
                    h1,h2,h2,h2,h1,h2,h2,h2,h1,
                    i1,i2,i2,i2,i1,i2,i2,i2,i1,
                    j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1,
                    k1,k2,k3,k1,k3,k2,k2,k1,k3,k2,k3,k1,k3,k1,k2,k3,k2,k1,
                    l1,l2,l3,l1,l3,l2,l2,l1,l3,l2,l3,l1,l3,l1,l2,l3,l2,l1,
                    m1,m2,m3,m1,m3,m2,m2,m1,m3,m2,m3,m1,m3,m1,m2,m3,m2,m1,
                    n1,n2,n3,n1,n3,n2,n2,n1,n3,n2,n3,n1,n3,n1,n2,n3,n2,n1,
                    o1,o2,o3,o1,o3,o2,o2,o1,o3,o2,o3,o1,o3,o1,o2,o3,o2,o1])

        W = array([wa,
                    wb,wb,wb,
                    wc,wc,wc,
                    wd,wd,wd,
                    we,we,we,
                    wf,wf,wf,
                    wg,wg,wg,
                    wh,wh,wh,
                    wi,wi,wi,
                    wj,wj,wj,wj,wj,wj,
                    wk,wk,wk,wk,wk,wk,
                    wl,wl,wl,wl,wl,wl,
                    wm,wm,wm,wm,wm,wm,
                    wn,wn,wn,wn,wn,wn,
                    wo,wo,wo,wo,wo,wo])


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
    surf.xi = average(surf.vertex[surf.triangle[:],0], axis=1)
    surf.yi = average(surf.vertex[surf.triangle[:],1], axis=1)
    surf.zi = average(surf.vertex[surf.triangle[:],2], axis=1)

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

#   Compute preconditioner
    computePrecond(surf)

    tic = time.time()
    sortPoints(surf, surf.tree, surf.twig, param)
    toc = time.time()
    time_sort = toc - tic

    return time_sort


def initializeField(filename, param):
    
    LorY, pot, E, kappa, charges, coulomb, qfile, Nparent, parent, Nchild, child = readFields(filename)

    Nfield = len(LorY)
    field_array = []
    Nchild_aux = 0
    for i in range(Nfield):
        if int(pot[i])==1:
            param.E_field.append(i)                                 # This field is where the energy will be calculated
        field_aux = fields()

        try:
            field_aux.LorY  = int(LorY[i])                              # Laplace of Yukawa
        except ValueError:
            field_aux.LorY  = 0
        try:
            field_aux.E     = param.REAL(E[i])                          # Dielectric constant
        except ValueError:
            field_aux.E  = 0
        try:
            field_aux.kappa = param.REAL(kappa[i])                      # inverse Debye length
        except ValueError:
            field_aux.kappa = 0

        field_aux.coulomb = int(coulomb[i])                         # do/don't coulomb interaction
        if int(charges[i])==1:                                      # if there are charges
            if qfile[i][-4:]=='.crd':
                xq,q,Nq = readcrd(qfile[i], param.REAL)             # read charges
                print '\nReading crd for region %i from '%i+qfile[i]
            if qfile[i][-4:]=='.pqr':
                xq,q,Nq = readpqr(qfile[i], param.REAL)             # read charges
                print '\nReading pqr for region %i from '%i+qfile[i]
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

    files, surf_type, phi0_file = readSurf(filename)      # Read filenames for surfaces
    Nsurf = len(files)

    for i in range(Nsurf):
        print '\nReading surface %i from file '%i + files[i]

        s = surfaces()

        s.surf_type = surf_type[i]

        if s.surf_type=='dirichlet_surface' or s.surf_type=='neumann_surface':
            s.phi0 = loadtxt(phi0_file[i])
            print '\nReading phi0 file for surface %i from '%i+phi0_file[i]

        Area_null = []
        tic = time.time()
        s.vertex = readVertex(files[i]+'.vert', param.REAL)
        triangle_raw = readTriangle(files[i]+'.face', s.surf_type)
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
                    s.Ein      = field_array[j].E
                    s.LorY_in  = field_array[j].LorY
            if len(field_array[j].child)>0:
                if i in field_array[j].child:                # Outside region
                    s.kappa_out = field_array[j].kappa
                    s.Eout      = field_array[j].E
                    s.LorY_out  = field_array[j].LorY

        if s.surf_type!='dirichlet_surface' and s.surf_type!='neumann_surface':
            s.E_hat = s.Ein/s.Eout
        else:
            s.E_hat = 1

        s.dphi = zeros(len(s.triangle))
        s.phi = zeros(len(s.triangle))

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
        surf_array[s].sglInt_intDev = gpuarray.to_gpu(surf_array[s].sglInt_intSort.astype(REAL))
        surf_array[s].sglInt_extDev = gpuarray.to_gpu(surf_array[s].sglInt_extSort.astype(REAL))
        surf_array[s].vertexDev  = gpuarray.to_gpu(ravel(surf_array[s].vertex[surf_array[s].triangleSort]).astype(REAL))
        surf_array[s].xcDev      = gpuarray.to_gpu(ravel(surf_array[s].xcSort.astype(REAL)))
        surf_array[s].ycDev      = gpuarray.to_gpu(ravel(surf_array[s].ycSort.astype(REAL)))
        surf_array[s].zcDev      = gpuarray.to_gpu(ravel(surf_array[s].zcSort.astype(REAL)))
        
#       Avoid transferring size 1 arrays to GPU (some systems crash)
        Nbuff = 5
        if len(surf_array[s].sizeTarget)<Nbuff:
            sizeTarget_buffer = zeros(Nbuff, dtype=int32)    
            sizeTarget_buffer[:len(surf_array[s].sizeTarget)] = surf_array[s].sizeTarget[:]    
            surf_array[s].sizeTarDev = gpuarray.to_gpu(sizeTarget_buffer)
        else:
            surf_array[s].sizeTarDev = gpuarray.to_gpu(surf_array[s].sizeTarget.astype(int32))
       
#        surf_array[s].sizeTarDev = gpuarray.to_gpu(surf_array[s].sizeTarget.astype(int32))
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



def fill_phi(phi, surf_array):
# Places the result vector on surf structure 

    s_start = 0
    for i in range(len(surf_array)):
        s_size = len(surf_array[i].xi)
        if surf_array[i].surf_type=='dirichlet_surface':
            surf_array[i].phi = surf_array[i].phi0
            surf_array[i].dphi = phi[s_start:s_start+s_size]
            s_start += s_size
        elif surf_array[i].surf_type=='neumann_surface':
            surf_array[i].dphi = surf_array[i].phi0
            surf_array[i].phi  = phi[s_start:s_start+s_size]
            s_start += s_size
        elif surf_array[i].surf_type=='asc_surface':
            surf_array[i].dphi = phi[s_start:s_start+s_size]/surf_array[i].Ein
            surf_array[i].phi  = zeros(s_size)
            s_start += s_size
        else:
            surf_array[i].phi  = phi[s_start:s_start+s_size]
            surf_array[i].dphi = phi[s_start+s_size:s_start+2*s_size]
            s_start += 2*s_size

        




