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
from math import atan2
import sys
sys.path.append('../utilities')
from util  import *
from util_arr import calculate_gamma as calculate_gamma_arr
from util_arr import test_pos as test_pos_arr

# y: triangle vertices
# x: source point
# same: 1 if x is in triangle, 0 if not
def AI(y, x, same):

    eps = 1e-16
    L     = array([y[1]-y[0], y[2]-y[1], y[0]-y[2]])

    Lu     = array([L[0]/norm(L[0]), L[1]/norm(L[1]), L[2]/norm(L[2])]) # Unit vectors parallel to L0, L1 and L2

    normal = cross(L[0],L[2])
    normal = normal/norm(normal)

    theta = zeros(3)
    theta[0] = arccos(dot(Lu[0],-Lu[2]))
    theta[1] = arccos(dot(Lu[1],-Lu[0]))
    theta[2] = pi - theta[0] - theta[1]

    tu = array([cross(normal,Lu[0]),cross(normal,Lu[1]),cross(normal,Lu[2])]) # Unit vector in triangle plane normal to L0, L1 and L2

    etha = dot(x-y[0], normal)
    rho = array([norm(x-y[0]), norm(x-y[1]), norm(x-y[2])])

    pp = zeros((3))
    pp = [dot(x-y[0],Lu[0]), dot(x-y[0],Lu[1]), dot(x-y[0],Lu[2])] # Distance x to y0 projected in L0, L1, and L2

    p = zeros((3,3))
    p[0] = [dot(x-y[0],Lu[0]), dot(x-y[0],Lu[1]), dot(x-y[0],Lu[2])] # Distance x to y0 projected in L0, L1, and L2
    p[1] = [dot(x-y[1],Lu[0]), dot(x-y[1],Lu[1]), dot(x-y[1],Lu[2])] # Distance x to y1 projected in L0, L1, and L2
    p[2] = [dot(x-y[2],Lu[0]), dot(x-y[2],Lu[1]), dot(x-y[2],Lu[2])] # Distance x to y2 projected in L0, L1, and L2
    p = -p # Don't really know why!

    q = zeros(3)
    q[0] = dot(x-y[0],tu[0])
    q[1] = dot(x-y[1],tu[1])
    q[2] = dot(x-y[2],tu[2])

    gamma = calculate_gamma(p, q, rho, etha)

    rhop1 = array([norm(x-y[1]), norm(x-y[2]), norm(x-y[0])])
    pp1      = array([p[1,0], p[2,1], p[0,2]])

    chi = log(diag(p) + rho) - log(pp1 + rhop1)

    aQ = dot(y[0]-y[2],tu[0])
    bQ = dot(y[1]-y[0],Lu[0])
    cQ = dot(y[2]-y[0],Lu[0])

    theta0 = test_pos(aQ, bQ, cQ, q, p, same)

    if (etha<1e-10): THETA = 0.5*sum(gamma) - theta0
    else: THETA = 0.5*sum(gamma) + theta0

    Q = dot(q,chi) - etha*THETA
    H = -THETA

    return H, Q


def AI_arr(DorN,y, x, same, E_hat):
    # y: array of triangles

    eps = 1e-16

    # Vector paralle to each side of triangle
    L = zeros((len(y),3,3))
    L[:,0] = y[:,1]-y[:,0]
    L[:,1] = y[:,2]-y[:,1]
    L[:,2] = y[:,0]-y[:,2]

    # Unit vectors parallel to L[0], L[1] and L[2]
    Lu = zeros((len(y),3,3))
    normL = sqrt(sum(L[:,0]**2,axis=1))
    Lu[:,0] = L[:,0]/transpose(normL*ones((3,len(y))))
    normL = sqrt(sum(L[:,1]**2,axis=1))
    Lu[:,1] = L[:,1]/transpose(normL*ones((3,len(y))))
    normL = sqrt(sum(L[:,2]**2,axis=1))
    Lu[:,2] = L[:,2]/transpose(normL*ones((3,len(y))))

    # Normal vector to panels
    normal = zeros((len(y),3))
    normal[:] = cross(L[:,0],L[:,2])
    norm_normal = sqrt(sum(normal**2,axis=1))
    normal = normal/transpose(norm_normal*ones((3,len(y))))

    theta = zeros((len(y),3))
    theta[:,0] = arccos(sum(Lu[:,0]*-Lu[:,2],axis=1))
    theta[:,1] = arccos(sum(Lu[:,1]*-Lu[:,0],axis=1))
    theta[:,2] = pi - theta[:,0] - theta[:,1]

    # Unit vector in triangle plane normal to L0, L1 and L2
    tu = zeros((len(y),3,3))
    tu[:,0] = cross(normal[:],Lu[:,0])
    tu[:,1] = cross(normal[:],Lu[:,1])
    tu[:,2] = cross(normal[:],Lu[:,2])

    # etha = x-y[0] \cdot normal
    etha = sum((x-y[:,0])*normal,axis=1)

    # rho = norm(x-y[0]), norm(x-y[1]), norm(x-y[2])
    rho = zeros((len(y),3))
    rho[:,0] = sqrt(sum((x-y[:,0])**2,axis=1))
    rho[:,1] = sqrt(sum((x-y[:,1])**2,axis=1))
    rho[:,2] = sqrt(sum((x-y[:,2])**2,axis=1))

    p00 = -sum((x-y[:,0])*Lu[:,0], axis=1)
    p11 = -sum((x-y[:,1])*Lu[:,1], axis=1)
    p22 = -sum((x-y[:,2])*Lu[:,2], axis=1)
    p10 = -sum((x-y[:,1])*Lu[:,0], axis=1)
    p21 = -sum((x-y[:,2])*Lu[:,1], axis=1)
    p02 = -sum((x-y[:,0])*Lu[:,2], axis=1)

    q = zeros((len(y),3))
    q[:,0] = sum((x-y[:,0])*tu[:,0],axis=1)
    q[:,1] = sum((x-y[:,1])*tu[:,1],axis=1)
    q[:,2] = sum((x-y[:,2])*tu[:,2],axis=1)

    gamma = calculate_gamma_arr(p00, p11, p22, p10, p21, p02, q, rho, etha)

    rhop1 = zeros((len(y),3))
    rhop1[:,0] = sqrt(sum((x-y[:,1])**2,axis=1))
    rhop1[:,1] = sqrt(sum((x-y[:,2])**2,axis=1))
    rhop1[:,2] = sqrt(sum((x-y[:,0])**2,axis=1))

    pp1 = zeros((len(y),3))
    pp1[:,0] = p10
    pp1[:,1] = p21
    pp1[:,2] = p02

    chi = log(transpose(array([p00,p11,p22])) + rho) - log(pp1 + rhop1)

    aQ = sum((y[:,0]-y[:,2])*tu[:,0], axis=1)
    bQ = sum((y[:,1]-y[:,0])*Lu[:,0], axis=1)
    cQ = sum((y[:,2]-y[:,0])*Lu[:,0], axis=1)
    theta0 = test_pos_arr(aQ, bQ, cQ, q, p00, same)

    THETA = where(etha<1e-10, 0.5*sum(gamma,axis=1)-theta0, 0.5*sum(gamma,axis=1)+theta0)

    G  = sum(q*chi,axis=1) - etha*THETA
    dG = -THETA

    E_HAT = where(same==1, E_hat, 1.)
    DIAG = where(same==1, 0., 1.)  # make diagonal of dG==0

    """
    print 'chi'
    print chi 
    print 'theta0'
    print theta0
    print 'p00 p10 p11 p21 p02 p22'
    print p00, p10, p11, p21, p02, p22
    print 'q'
    print q
    print 'gamma'
    print gamma
    print 'THETA'
    print THETA
    print 'omega'
    print dot(q[0],chi[0])
    """

    if DorN==0:
        return G*E_HAT
    elif DorN==1:
        return dG
    else:
        return G*E_HAT, dG*DIAG

def GQ(xi, x):
    r = norm(x-xi)
    r3 = r**3
    G  = 1/r
    Hx = (xi-x)[0]/r3
    Hy = (xi-x)[1]/r3
    Hz = (xi-x)[2]/r3

    return Hx,Hy,Hz,G


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
