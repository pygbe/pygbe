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
    
    if (p1<0. and p2>0.):
        
        if qet<0.:
            cond = (cr1>0. and cr2<0. and zn>0. and zd<0.) or (cr1<0. and cr2>0. and zn>0. and zd<0.) or (cr1<0. and cr2<0. and zn>0.)
            if cond:
                return atan2(zn,zd) - 2*pi
            else:
                return atan2(zn,zd)

        elif qet>0.:
            cond = (cr1>0. and cr2<0. and zn<0. and zd<0.) or (cr1<0. and cr2>0. and zn<0. and zd<0.) or (cr1<0. and cr2<0. and zn<0.)
            if cond:
                return atan2(zn,zd)+2*pi
            else:
                return atan2(zn,zd)

        else:
            return 0.0

    else:
        return atan2(zn,zd)

                 

def calculate_gamma(p, q, rho, etha):
    
    cr11 = q[0]**2*rho[0]**2 - p[0,0]**2*etha**2 
    cr12 = q[0]**2*rho[1]**2 - p[1,0]**2*etha**2 
    zn1  = 2*q[0]*etha*(p[1,0]*rho[1]*cr11 - p[0,0]*rho[0]*cr12)
    zd1  = cr11*cr12 + (2*q[0]*etha)**2 * p[0,0]*rho[0]*p[1,0]*rho[1]
    
    cr22 = q[1]**2*rho[1]**2 - p[1,1]**2*etha**2 
    cr23 = q[1]**2*rho[2]**2 - p[2,1]**2*etha**2 
    zn2  = 2*q[1]*etha*(p[2,1]*rho[2]*cr22 - p[1,1]*rho[1]*cr23)
    zd2  = cr22*cr23 + (2*q[1]*etha)**2 * p[1,1]*rho[1]*p[2,1]*rho[2]
    
    cr33 = q[2]**2*rho[2]**2 - p[2,2]**2*etha**2 
    cr31 = q[2]**2*rho[0]**2 - p[0,2]**2*etha**2 
    zn3  = 2*q[2]*etha*(p[0,2]*rho[0]*cr33 - p[2,2]*rho[2]*cr31)
    zd3  = cr33*cr31 + (2*q[2]*etha)**2 * p[2,2]*rho[2]*p[0,2]*rho[0]

    #print(cr11, cr12, zn1, zd1)
    #print(cr22, cr23, zn2, zd2)
    #print(cr33, cr31, zn3, zd3)

    gamma = zeros(3)
    gamma[0] = get_gamma(p[0,0],p[1,0],2*q[0]*etha,cr11,cr12,zn1,zd1)
    gamma[1] = get_gamma(p[1,1],p[2,1],2*q[1]*etha,cr22,cr23,zn2,zd2)
    gamma[2] = get_gamma(p[2,2],p[0,2],2*q[2]*etha,cr33,cr31,zn3,zd3)

    return gamma

def test_pos(aQ, bQ, cQ, q, p, same):

    if same: return 2*pi
    
    kQ3 = cQ/aQ
    shc = zeros(3)
    shc[2] = -q[0]/aQ
    shc[1] = (kQ3*q[0] - p[0,0])/bQ
    shc[0] = 1. - (shc[2] + shc[1])

    if (shc[0]>1e-12 and shc[1]>1e-12 and shc[2]>1e-12): 
        return 2*pi
    elif (abs(shc[0])<1e-12 and shc[1]>0. and shc[2]>0.):
        return pi
    elif (abs(shc[1])<1e-12 and shc[2]>0. and shc[2]<1.):
        return pi
    elif (abs(shc[2])<1e-12 and shc[1]>0. and shc[1]<1.):
        return pi
    else:
        return 0.
    
