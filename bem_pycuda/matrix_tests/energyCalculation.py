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
from GaussIntegration import gaussIntegration_fine


def calculate_phir(phi, dphi, s, xq, K_fine, eps, LorY, kappa):

    phir = 0
    dummy = array([[0,0,0]])
    for i in range(len(s.triangle)):
        panel = s.vertex[s.triangle[i]]

        K, V, Kp = gaussIntegration_fine(xq, panel, s.normal[i], s.Area[i], dummy, K_fine, kappa, LorY, eps)
        # s.normal is dummy: needed for Kp, which we don't use here.
        phir += (-K*phi[i] + V*dphi[i])/(4*pi)
        
    return phir

def solvationEnergy(surf_array, field_array, param):

    Esolv = []
    field_Esolv = []
    cal2J = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(cal2J*param.E_0)

    for i in range(len(field_array)):
        f = field_array[i]
        if len(f.xq)>0:
        
            phi_reac = 0

#           First look at child surfaces
            for cs in f.child:
                s = surf_array[cs]
                phi_aux = calculate_phir(s.phi, s.Ehat*s.dphi, s, f.xq, param.K_fine, param.eps, f.LorY, f.kappa)

                phi_reac -= phi_aux     # Minus accounts for normals pointing out
            
#           Now look at parent surface
            if len(f.parent)>0:
                ps = f.parent[0]
                s = surf_array[ps] 
                phi_aux = calculate_phir(s.phi, s.dphi, s, f.xq, param.K_fine, param.eps, f.LorY, f.kappa)
                
                phi_reac += phi_aux

            Esolv.append(0.5*C0*sum(f.q*phi_reac))
            field_Esolv.append(i)

    return Esolv, field_Esolv

def coulombicEnergy(field_array, param):

    Ecoul = []
    field_coul = []
    cal2J = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(cal2J*param.E_0)
    for i in range(len(field_array)):
        f = field_array[i]
        if f.coul==1:
            Nq = len(f.xq)
            x = f.xq[:,0]
            y = f.xq[:,1]
            z = f.xq[:,2]
            
            dx = transpose(ones(Nq,Nq)*x) - x
            dy = transpose(ones(Nq,Nq)*y) - y
            dz = transpose(ones(Nq,Nq)*z) - z

            r = sqrt(dx*dx+dy*dy+dz*dz)

            M = 1/(f.E*r)

            phi_q = dot(M,f.q)

            Ecoul.append(0.5*C0*sum(phi_q*f.q))
            field_coul.append(i)

    return Ecoul, field_coul

    


def surfaceEnergy(surf_array, param):

    Esurf = []
    surf_Esurf = []
    cal2J = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(cal2J*param.E_0)

    for i in range(len(surf_array)):
        s = surf_array[i]

        if s.surf_type == 'dirichlet_surface':
            surf_Esurf.append(i)
            Esurf_aux = -sum(-s.Eout*s.dphi*s.phi*s.Area)
            Esurf.append(0.5*C0*Esurf_aux)
            
        elif s.surf_type == 'neumann_surface':
            surf_Esurf.append(i)
            Esurf_aux = sum(-s.Eout*s.dphi*s.phi*s.Area)
            Esurf.append(0.5*C0*Esurf_aux)

    return Esurf, surf_Esurf

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
        else:
            surf_array[i].phi  = phi[s_start:s_start+s_size]
            surf_array[i].dphi = phi[s_start+s_size:s_start+2*s_size]
            s_start += 2*s_size

def dipoleMoment(surf_array, electricField):
# Computes dipole moment on a surface
# Dipole is expressed as a boundary integral

    for i in range(len(surf_array)):
        s = surf_array[i]

        xc = array([s.xi, s.yi, s.zi])

#       Change dphi to outer side of surface
        dphi = s.dphi*s.Ehat - (1-s.Ehat)*electricField*s.normal[:,2]

        I1 = sum(xc*dphi*s.Area, axis=1)
        I2 = sum(transpose(s.normal)*s.phi*s.Area, axis=1)

        s.dipole = s.Eout*(I1-I2)

def extCrossSection(surf_array, k, n, wavelength, electricField):
# Computes the extinction cross section (According to Mischenko2007)
# k: unit vector in direction of wave propagation
# n: unit vector in direction of electric field

    Cext = []
    surf_Cext = []
    for i in range(len(surf_array)):
        s = surf_array[i]

        diffractionCoeff = sqrt(s.Eout)
        waveNumber = 2*pi*diffractionCoeff/wavelength

        v1 = cross(k, s.dipole)
        v2 = cross(v1, k)

        C1 = dot(n, v2) * waveNumber**2/(s.Eout*electricField)

        Cext.append(1/waveNumber.real * C1.imag)
        surf_Cext.append(i)

    return Cext, surf_Cext
