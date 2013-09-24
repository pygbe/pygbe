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
#from pylab import *
from math import pi, atan2

def get_gamma(p1, p2, qet, cr1, cr2, zn, zd):
    
    cond_aux1 = logical_and(cr1>0.,logical_and(cr2<0.,zd<0.))
    cond_aux2 = logical_and(cr1<0.,logical_and(cr2>0.,zd<0.))
    cond1 = logical_or(logical_and(cond_aux1,zn>0.),logical_or(logical_and(cond_aux2,zn>0.),logical_and(cr1<0.,logical_and(cr2<0.,zn>0.))))
    cond2 = logical_or(logical_and(cond_aux1,zn<0.),logical_or(logical_and(cond_aux2,zn<0.),logical_and(cr1<0.,logical_and(cr2<0.,zn<0.))))

    aTan = arctan2(zn,zd)
    gm = arctan2(zn,zd)

    cond_aux3 = logical_and(p1<0,p2>0)
    C1 = logical_and(cond_aux3,logical_and(qet<0,cond1))
    C2 = logical_and(cond_aux3,logical_and(qet>0,cond2))
    C3 = logical_and(cond_aux3,qet==0.)

    gm = where(C1, aTan-2*pi, gm)
    gm = where(logical_and(C2,logical_not(C1)), aTan+2*pi, gm)
    gm = where(logical_and(C3, logical_and(logical_not(C1),logical_not(C2))), zeros(len(gm)), gm)

    return gm

def calculate_gamma(p00, p11, p22, p10, p21, p02, q, rho, etha):
    
    cr11 = q[:,0]**2*rho[:,0]**2 - p00**2*etha**2 
    cr12 = q[:,0]**2*rho[:,1]**2 - p10**2*etha**2 
    zn1  = 2*q[:,0]*etha*(p10*rho[:,1]*cr11 - p00*rho[:,0]*cr12)
    zd1  = cr11*cr12 + (2*q[:,0]*etha)**2 * p00*rho[:,0]*p10*rho[:,1]
    
    cr22 = q[:,1]**2*rho[:,1]**2 - p11**2*etha**2 
    cr23 = q[:,1]**2*rho[:,2]**2 - p21**2*etha**2 
    zn2  = 2*q[:,1]*etha*(p21*rho[:,2]*cr22 - p11*rho[:,1]*cr23)
    zd2  = cr22*cr23 + (2*q[:,1]*etha)**2 * p11*rho[:,1]*p21*rho[:,2]
    
    cr33 = q[:,2]**2*rho[:,2]**2 - p22**2*etha**2 
    cr31 = q[:,2]**2*rho[:,0]**2 - p02**2*etha**2 
    zn3  = 2*q[:,2]*etha*(p02*rho[:,0]*cr33 - p22*rho[:,2]*cr31)
    zd3  = cr33*cr31 + (2*q[:,2]*etha)**2 * p22*rho[:,2]*p02*rho[:,0]

    #print(cr11, cr12, zn1, zd1)
    #print(cr22, cr23, zn2, zd2)
    #print(cr33, cr31, zn3, zd3)

    gamma = zeros((len(p00),3))
    gamma[:,0] = get_gamma(p00,p10,2*q[:,0]*etha,cr11,cr12,zn1,zd1)
    gamma[:,1] = get_gamma(p11,p21,2*q[:,1]*etha,cr22,cr23,zn2,zd2)
    gamma[:,2] = get_gamma(p22,p02,2*q[:,2]*etha,cr33,cr31,zn3,zd3)

    return gamma

def test_pos(aQ, bQ, cQ, q, p00, same):

    kQ3 = cQ/aQ
    shc = zeros((len(p00),3))
    shc[:,2] = -q[:,0]/aQ
    shc[:,1] = (kQ3*q[:,0] - p00)/bQ
    shc[:,0] = 1. - (shc[:,2] + shc[:,1])

    # Condition if outside triangle
    cond = logical_or(same==1, logical_and(shc[:,0]>1e-15, logical_and(shc[:,1]>1e-15, shc[:,2]>1e-15)))
    position = where(cond, ones(len(p00))*2*pi, zeros(len(p00)))

    # Condition if on any edge   
    # Edge 2
    cond = logical_and(abs(shc[:,0])<1e-12, logical_and(shc[:,1]>0., shc[:,2]>0.))
    position = where(cond, ones(len(p00))*pi, position)
    # Edge 3
    cond = logical_and(abs(shc[:,1])<1e-12, logical_and(shc[:,2]>0., shc[:,2]<1.))
    position = where(cond, ones(len(p00))*pi, position)
    # Edge 1
    cond = logical_and(abs(shc[:,2])<1e-12, logical_and(shc[:,1]>0., shc[:,1]<1.))
    position = where(cond, ones(len(p00))*pi, position)


    return position 
