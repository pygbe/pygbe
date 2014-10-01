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

def charge2surf(s, xq, q, E): 
    
    Nq = len(xq)
    N  = len(s.xi)
    dx_pq = zeros((Nq,N))
    dy_pq = zeros((Nq,N))
    dz_pq = zeros((Nq,N))
    for i in range(Nq):
        dx_pq[i,:] = xq[i,0] - s.xi
        dy_pq[i,:] = xq[i,1] - s.yi
        dz_pq[i,:] = xq[i,2] - s.zi

    R_pq  = sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)
    F = sum(transpose(q*ones((N,Nq)))/(E*R_pq),axis=0)

    return F

def generateRHS(surf_array, field_array, Neq, ElectricField=0.):

#   Check if there is a complex dielectric
    complexDiel = 0
    for f in field_array:
        if type(f.E)==complex:
            complexDiel = 1

    if complexDiel==1:
        F = zeros(Neq, complex)
    else:
        F = zeros(Neq)

    F_sym = []
    X_sym = []
    Nblock = 0
    for s in surf_array:
        if s.surf_type=='dirichlet_surface' or s.surf_type=='neumann_surface':
            F_sym.append([''])
            X_sym.append([''])
            Nblock += 1
        else:
            F_sym.append(['',''])
            X_sym.append(['',''])
            Nblock += 2

    for f in field_array:
#       Effect of charges
        if len(f.xq)>0:

#           Children first
            if len(f.child)>0:
                for i in f.child:
                    s = surf_array[i]
                    if s.surf_type=='dirichlet_surface' or s.surf_type=='neumann_surface':
                        F[s.N0:s.N0+s.N] += charge2surf(s, f.xq, f.q, f.E)
                        F_sym[i][0] += '       sum_%i'%i
                    else:
                        F[s.N0+s.N:s.N0+2*s.N] += charge2surf(s, f.xq, f.q, f.E)
                        F_sym[i][1] += '       sum_%i'%i

#           Parent surface
            if len(f.parent)>0:
                i = f.parent[0]
                s = surf_array[i]
                F[s.N0:s.N0+s.N] += charge2surf(s, f.xq, f.q, f.E)
                F_sym[i][0] += '       sum_%i'%i


#       Effect of charged surfaces
#       Only child surfaces can be charged
        for j in f.child:
            src = surf_array[j]
#           Dirichlet surfaces
            if src.surf_type=='dirichlet_surface':
                X_sym[j][0] += 'dphi%i'%j
#               On child surfaces (exterior equation)
                for i in f.child:
                    tar = surf_array[i]
#                   Dirichlet and neumann surface have only one equation
                    if tar.surf_type=='dirichlet_surface' or tar.surf_type=='neumann_surface':
                        F[tar.N0:tar.N0+tar.N] += dot(-tar.Kext[j], src.phi0)
                        F_sym[i][0] += '-'+tar.KextSym[j]
#                   Rest have two equations: put in exterior
                    else:
                        F[tar.N0+tar.N:tar.N0+2*tar.N] += dot(-tar.Kext[j], src.phi0)
                        F_sym[i][1] += '-'+tar.KextSym[j]
#               On parent surface (interior equation)
                if len(f.parent)>0:
                    i = f.parent[0]
                    tar = surf_array[i]
                    F[tar.N0:tar.N0+tar.N] += dot(-tar.Kint[j], src.phi0)
                    F_sym[i][0] += '-'+tar.KintSym[j]

            
#           Neumann surfaces
            elif src.surf_type=='neumann_surface':
                X_sym[j][0] += ' phi%i'%j
#               On child surfaces (exterior equation)
                for i in f.child:
                    tar = surf_array[i]
#                   Dirichlet and neumann surface have only one equation
                    if tar.surf_type=='dirichlet_surface' or tar.surf_type=='neumann_surface':
                        F[tar.N0:tar.N0+tar.N] += dot(-tar.Vext[j], src.phi0)
                        F_sym[i][0] += '-'+tar.VextSym[j]
#                   Rest have two equations: put in exterior
                    else:
                        F[tar.N0+tar.N:tar.N0+2*tar.N] += dot(-tar.Vext[j], src.phi0)
                        F_sym[i][1] += '-'+tar.VextSym[j]
#               On parent surface (interior equation)
                if len(f.parent)>0:
                    i = f.parent[0]
                    tar = surf_array[i]
                    F[tar.N0:tar.N0+tar.N] += dot(-tar.Vint[j], src.phi0)
                    F_sym[i][0] += '-'+tar.VintSym[j]

            else:
                X_sym[j][0] += ' phi%i'%j
                X_sym[j][1] += 'dphi%i'%j

#       Effect of an incomming electric field (only on outmost region)
#       Field comes in the z direction
        if len(f.parent)==0 and abs(ElectricField)>1e-12:

#           On child surfaces (exterior equation)
            for i in f.child:
                tar = surf_array[i]

#               TO BE IMPLEMENTED: Dirichlet and neumann surface have only one equation
                if tar.surf_type=='dirichlet_surface' or tar.surf_type=='neumann_surface':
                    pass
#                    F[tar.N0:tar.N0+tar.N] += 0 #dot(-tar.Kext[j], src.phi0)
#                    F_sym[i][0] += 0 #'-'+tar.KextSym[j]

#               Rest have two equations: put in exterior
                else:
                    phi_field = ElectricField*tar.normal[:,2] #Assuming field comes in z direction
                    F[tar.N0+tar.N:tar.N0+2*tar.N] += (1/tar.Ehat-1) * dot(tar.Vext[i], phi_field)
                    F_sym[i][1] += tar.VextSym[i]+'_E(1/Eh-1)'

        

    for i in range(len(F_sym)):
        for j in range(len(F_sym[i])):
            if len(F_sym[i][j])==0:
                F_sym[i][j] = '           0'

        

    return F, F_sym, X_sym, Nblock
