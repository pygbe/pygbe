import numpy
from semi_analyticalwrap import SA_wrap_arr

def GQ_1D(K):
    T = numpy.zeros((K,K))
    nvec = numpy.arange(1.,K)
    beta = 0.5/numpy.sqrt(1-1/(2*nvec)**2)
    T = numpy.diag(beta,1)+numpy.diag(beta,-1)
    d,v = numpy.linalg.eig(T)
    w = 2*v[0]**2
    x = d

    return x,w

def lineInt(z, x, v1, v2, kappa, xk, wk):

    theta1 = numpy.arctan2(v1,x)
    theta2 = numpy.arctan2(v2,x)

    dtheta = theta2 - theta1

    absZ = abs(z)
    if absZ < 1e-10 : signZ = 0
    else            : signZ = z/absZ

    dtheta = theta2 - theta1
    thetam = (theta2+theta1)/2.

    thetak = dtheta/2*xk + thetam
    Rtheta = x/numpy.cos(thetak)
    dy = x*numpy.tan(thetak)
    R = numpy.sqrt(Rtheta**2+z**2)
    
    phi_Y  = numpy.sum(-wk * (exp(-kappa*R) - exp(-kappa*absZ))/kappa)
    dphi_Y = -numpy.sum(wk *(z/R*exp(-kappa*R) - exp(-kappa*absZ)*signZ))

    phi_L  = numpy.sum(wk * (R - absZ))
    dphi_L = -numpy.sum(wk *(z/R - signZ))

    phi_Y  *= dtheta/2
    dphi_Y *= dtheta/2
    phi_L  *= dtheta/2
    dphi_L *= dtheta/2

    return phi_Y,dphi_Y,phi_L,dphi_L

def intSide(v1, v2, p, kappa, xk, wk):
    
    v21 = v2 - v1
    L21 = numpy.linalg.norm(v21)
    v21u = v21/L21
    orthog = numpy.cross(numpy.array([0,0,1]), v21u)

    alpha = -numpy.dot(v21,v1)/L21**2

    rOrthog = v1+alpha*v21
    d_toEdge = numpy.linalg.norm(rOrthog)
    side_vec = numpy.cross(v21,-v1)

    rotateToVertLine = numpy.zeros((3,3))
    rotateToVertLine[:,0] = orthog
    rotateToVertLine[:,1] = v21u
    rotateToVertLine[:,2] = [0.,0.,1.]

    v1new = numpy.dot(rotateToVertLine,v1)

    if v1new[0]<0:
        v21u = -v21u
        orthog = -orthog
        rotateToVertLine[:,0] = orthog
        rotateToVertLine[:,1] = v21u
        v1new = numpy.dot(rotateToVertLine,v1)

    v2new = numpy.dot(rotateToVertLine, v2)
    rOrthognew = numpy.dot(rotateToVertLine, rOrthog)
    x = v1new[0]

    if v1new[1]>0 and v2new[1]<0 or v1new[1]<0 and v2new[1]>0:
        phi1_Y, dphi1_Y, phi1_L, dphi1_L = lineInt(p, x, 0, v1new[1],kappa, xk, wk) 
        phi2_Y, dphi2_Y, phi2_L, dphi2_L = lineInt(p, x, v2new[1], 0,kappa, xk, wk)

        phi_Y  = phi1_Y+phi2_Y
        dphi_Y = dphi1_Y+dphi2_Y
        phi_L  = phi1_L+phi2_L
        dphi_L = dphi1_L+dphi2_L

    else:
        phi_Y, dphi_Y, phi_L, dphi_L = lineInt(p, x, v1new[1], v2new[1],kappa, xk, wk)
        phi_Y  = -phi_Y
        dphi_Y = -dphi_Y
        phi_L  = -phi_L
        dphi_L = -dphi_L

    return phi_Y, dphi_Y, phi_L, dphi_L

def SA_arr(y, x, kappa, same, xk, wk):
    
    N = len(x)
    phi_Y  = numpy.zeros(N)
    dphi_Y = numpy.zeros(N)
    phi_L  = numpy.zeros(N)
    dphi_L = numpy.zeros(N)

    # Put first vertex at origin
    y_panel = y - y[0]
    x_panel = x - y[0]

    # Find panel coordinate system X: 0->1
    X = y_panel[1]
    X = X/numpy.linalg.norm(X)
    Z = numpy.cross(y_panel[1],y_panel[2])
    Z = Z/numpy.linalg.norm(Z)
    Y = numpy.cross(Z,X)

    # Rotate coordinate system to match panel plane
    rot_matrix = numpy.array([X,Y,Z])
    panel_plane = numpy.transpose(numpy.dot(rot_matrix,numpy.transpose(y_panel)))
    x_plane = numpy.transpose(numpy.dot(rot_matrix, numpy.transpose(x_panel)))


    for i in range(N):
        # Shift origin so it matches collocation point
        panel_final = panel_plane - numpy.array([x_plane[i,0],x_plane[i,1],0])

        # Loop over sides
        for j in range(3):
            if j==2: nextJ = 0
            else:    nextJ = j+1
          
            phi_Y_aux, dphi_Y_aux, phi_L_aux, dphi_L_aux = intSide(panel_final[j], panel_final[nextJ], x_plane[i,2],kappa, xk, wk)
            phi_Y[i]  += phi_Y_aux
            dphi_Y[i] += dphi_Y_aux
            phi_L[i]  += phi_L_aux
            dphi_L[i] += dphi_L_aux

        if same[i]==1: 
            dphi_Y[i] = 2*pi
            dphi_L[i] = -2*pi
    
    return phi_Y, dphi_Y, phi_L, dphi_L

def GQ(y, x, kappa, same):
    # n=7
    L = numpy.array([y[1]-y[0], y[2]-y[1], y[0]-y[2]])
    Area = numpy.linalg.norm(cross(L[2],L[1]))/2
    normal = numpy.cross(L[0],L[2])
    normal = normal/numpy.linalg.norm(normal)

    M = numpy.transpose(y)
    xi = numpy.zeros((7,3))
    m  = numpy.zeros(7)
    xi[0] = numpy.dot(M,numpy.array([1/3.,1/3.,1/3.]))
    xi[1] =numpy.dot(M,numpy.array([.79742699,.10128651,.10128651]))
    xi[2] =numpy.dot(M,numpy.array([.10128651,.79742699,.10128651]))
    xi[3] =numpy.dot(M,numpy.array([.10128651,.10128651,.79742699]))
    xi[4] =numpy.dot(M,numpy.array([.05971587,.47014206,.47014206]))
    xi[5] =numpy.dot(M,numpy.array([.47014206,.05971587,.47014206]))
    xi[6] =numpy.dot(M,numpy.array([.47014206,.47014206,.05971587]))
    r = numpy.sqrt(numpy.sum((x-xi)**2,axis=1))

    m[0] = 0.225
    m[1] = 0.12593918
    m[2] = 0.12593918
    m[3] = 0.12593918
    m[4] = 0.13239415
    m[5] = 0.13239415
    m[6] = 0.13239415
    Q17 = Area * numpy.sum(m*numpy.exp(-kappa*r)/r)
    Q27 = Area * numpy.sum(-m*numpy.exp(-kappa*r)*(kappa+1/r)/r**2*numpy.dot(xi-x,normal))

    if same==1: Q27 = 2*pi
    return Q17, Q27

"""
y = array([[-0.38268343, 0.,-0.92387953],[ 0.,-0.38268343, -0.92387953],[0., 0., -1.]])
#y = array([[-sqrt(2)/2.,-sqrt(2)/2.,0.],[0.,-sqrt(2)/2.,sqrt(2)/2.],[0.,-1.,0.]])
#x = array([7/6.,7/6.,17/12.])
#x = array([4/3.,4/3.,4/3.])
x = array([[-0.544331053952, -0.544331053952, -0.544331053952],[average(y[:,0]), average(y[:,1]), average(y[:,2])]])
#x = array([-0.235702, -0.333333, -0.235702])
same = array([0,1], dtype=int32)
eps = 1e-16
kappa=1.5
xk,wk = GQ_1D(5)

#for i in range(len(x)):
#    Q1,Q2 = GQ(y,x[i],kappa,same)
#    print Q1,Q2

IY,dIY,IL,dIL = SA_arr(y,x,kappa,same,xk,wk)

WY = zeros(2)
dWY = zeros(2)
WL = zeros(2)
dWL = zeros(2)
y1D = ravel(y)
x1D = ravel(x)
SA_wrap_arr(y1D,x1D,WY,dWY,WL,dWL,kappa,same,xk,wk)
print IY, WY 
print dIY, dWY
print IL, WL
print dIL, dWL

#print 'Error G : %s'%(abs(Q1-I1)/I1)
#print 'Error dG: %s'%(abs(Q2-I2)/I2)
"""
