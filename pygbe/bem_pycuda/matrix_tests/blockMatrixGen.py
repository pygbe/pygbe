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

from numpy              import *
from class_definition   import *
from scipy.sparse       import *
import sys
sys.path.append('../util')
from triangulation          import *
from integral_matfree       import *
from semi_analyticalwrap    import SA_wrap_arr
from GaussIntegration       import gaussIntegration_fine


def blockMatrix2(tar, src, WK, kappa, threshold, LorY, xk, wk, K_fine, eps):
    
    Ns = len(src.xi)
    Nt = len(tar.xi)
    K  = len(WK)

    dx = transpose(ones((Ns*K,Nt))*tar.xi) - src.xj
    dy = transpose(ones((Ns*K,Nt))*tar.yi) - src.yj
    dz = transpose(ones((Ns*K,Nt))*tar.zi) - src.zj
    r = sqrt(dx*dx+dy*dy+dz*dz+eps*eps)

    dx = reshape(dx,(Nt,Ns,K))
    dy = reshape(dy,(Nt,Ns,K))
    dz = reshape(dz,(Nt,Ns,K))
    r  = reshape(r,(Nt,Ns,K))

    if LorY==1:   # if Laplace
#       Double layer 
        K_lyr = src.Area * (sum(WK/r**3*dx, axis=2)*src.normal[:,0]
                          + sum(WK/r**3*dy, axis=2)*src.normal[:,1]
                          + sum(WK/r**3*dz, axis=2)*src.normal[:,2])
#       Single layer
        V_lyr = src.Area * sum(WK/r, axis=2)

    else:           # if Yukawa
#       Double layer 
        K_lyr = src.Area * (sum(WK/r**2*exp(-kappa*r)*(kappa+1/r)*dx, axis=2)*src.normal[:,0]
                          + sum(WK/r**2*exp(-kappa*r)*(kappa+1/r)*dy, axis=2)*src.normal[:,1]
                          + sum(WK/r**2*exp(-kappa*r)*(kappa+1/r)*dz, axis=2)*src.normal[:,2])
#       Single layer
        V_lyr = src.Area * sum(WK * exp(-kappa*r)/r, axis=2)
        

    same = zeros((Nt,Ns),dtype=int32)
    if abs(src.xi[0]-tar.xi[0])<1e-10:
        for i in range(Nt):
            same[i,i]   = 1 

    L_d  = greater_equal(sqrt(2*src.Area)/average(r,axis=2),threshold)

    N_analytical = 0 

    for i in range(Ns):
        panel = src.vertex[src.triangle[i]]
        an_integrals = nonzero(L_d[:,i])[0]
        local_center = zeros((len(an_integrals),3))
        local_center[:,0] = tar.xi[an_integrals]
        local_center[:,1] = tar.yi[an_integrals]
        local_center[:,2] = tar.zi[an_integrals]

        G_Y  = zeros(len(local_center))
        dG_Y = zeros(len(local_center))
        G_L  = zeros(len(local_center))
        dG_L = zeros(len(local_center))
        SA_wrap_arr(ravel(panel), ravel(local_center), G_Y, dG_Y, G_L, dG_L, kappa, same[an_integrals,i], xk, wk) 

        if LorY==1:   # if Laplace
            K_lyr[an_integrals,i] = dG_L
            V_lyr[an_integrals,i] = G_L
        else:           # if Yukawa
            K_lyr[an_integrals,i] = dG_Y
            V_lyr[an_integrals,i] = G_Y  

        N_analytical += len(an_integrals)

    print '\t%i analytical integrals'%(N_analytical/Ns)

    return K_lyr, V_lyr


def blockMatrix(tar, src, WK, kappa, threshold, LorY, xk, wk, K_fine, eps):
    
    Ns = len(src.xi)
    Nt = len(tar.xi)
    K  = len(WK)

    dx = transpose(ones((Ns*K,Nt))*tar.xi) - src.xj
    dy = transpose(ones((Ns*K,Nt))*tar.yi) - src.yj
    dz = transpose(ones((Ns*K,Nt))*tar.zi) - src.zj
    r = sqrt(dx*dx+dy*dy+dz*dz+eps*eps)

    dx = reshape(dx,(Nt,Ns,K))
    dy = reshape(dy,(Nt,Ns,K))
    dz = reshape(dz,(Nt,Ns,K))
    r  = reshape(r,(Nt,Ns,K))

    if LorY==1:   # if Laplace
#       Double layer 
        K_lyr = src.Area * (sum(WK/r**3*dx, axis=2)*src.normal[:,0]
                          + sum(WK/r**3*dy, axis=2)*src.normal[:,1]
                          + sum(WK/r**3*dz, axis=2)*src.normal[:,2])
#       Single layer
        V_lyr = src.Area * sum(WK/r, axis=2)
#       Adjoint double layer
        Kp_lyr = -src.Area * ( transpose(transpose(sum(WK/r**3*dx, axis=2))*tar.normal[:,0])
                             + transpose(transpose(sum(WK/r**3*dy, axis=2))*tar.normal[:,1])
                             + transpose(transpose(sum(WK/r**3*dz, axis=2))*tar.normal[:,2]) )

    else:           # if Yukawa
#       Double layer 
        K_lyr = src.Area * (sum(WK/r**2*exp(-kappa*r)*(kappa+1/r)*dx, axis=2)*src.normal[:,0]
                          + sum(WK/r**2*exp(-kappa*r)*(kappa+1/r)*dy, axis=2)*src.normal[:,1]
                          + sum(WK/r**2*exp(-kappa*r)*(kappa+1/r)*dz, axis=2)*src.normal[:,2])
#       Single layer
        V_lyr = src.Area * sum(WK * exp(-kappa*r)/r, axis=2)
#       Adjoint double layer
        Kp_lyr = zeros(shape(K_lyr))      #TO BE IMPLEMENTED
        

    same = zeros((Nt,Ns),dtype=int32)
    if abs(src.xi[0]-tar.xi[0])<1e-10:
        for i in range(Nt):
            same[i,i]   = 1 

    tri_ctr = average(src.vertex[src.triangle[:]], axis=1)
    dx = transpose(ones((Ns,Nt))*tar.xi) - tri_ctr[:,0]
    dy = transpose(ones((Ns,Nt))*tar.yi) - tri_ctr[:,1]
    dz = transpose(ones((Ns,Nt))*tar.zi) - tri_ctr[:,2]
    r_tri = sqrt(dx*dx+dy*dy+dz*dz)
    L_d  = logical_and(greater_equal(sqrt(2*src.Area)/(r_tri+eps),threshold), same==0) 

#    L_d  = logical_and(greater_equal(sqrt(2*src.Area)/average(r,axis=2),threshold), same==0) 

    N_analytical = 0 

    for i in range(Ns):
        panel = src.vertex[src.triangle[i]]
        an_integrals = nonzero(L_d[:,i])[0]
        local_center = zeros((len(an_integrals),3))
        local_center[:,0] = tar.xi[an_integrals]
        local_center[:,1] = tar.yi[an_integrals]
        local_center[:,2] = tar.zi[an_integrals]
        normal_tar = tar.normal[an_integrals]

        K_aux, V_aux, Kp_aux = gaussIntegration_fine(local_center, panel, src.normal[i], src.Area[i], normal_tar, K_fine, kappa, LorY, eps)
        K_lyr[an_integrals,i]  = K_aux[:,0]
        V_lyr[an_integrals,i]  = V_aux[:,0]
        Kp_lyr[an_integrals,i] = Kp_aux[:,0]

        N_analytical += len(an_integrals)

        if abs(src.xi[0]-tar.xi[0])<1e-10:
            if same[i,i] == 1:
                local_center = array([tar.xi[i], tar.yi[i], tar.zi[i]])
                G_Y  = zeros(1)
                dG_Y = zeros(1)
                G_L  = zeros(1)
                dG_L = zeros(1)
                SA_wrap_arr(ravel(panel), local_center, G_Y, dG_Y, G_L, dG_L, kappa, array([1], dtype=int32), xk, wk) 

                if LorY==1:   # if Laplace
                    K_lyr[i,i]  = dG_L
                    V_lyr[i,i]  = G_L
                    Kp_lyr[i,i] = dG_L
                else:           # if Yukawa
                    K_lyr[i,i]  = dG_Y
                    V_lyr[i,i]  = G_Y  
                    Kp_lyr[i,i] = dG_Y

                N_analytical += 1

    print '\t%i analytical integrals'%(N_analytical/Ns)

    return K_lyr, V_lyr, Kp_lyr


def generateMatrix(surf_array, Neq):
    
#   Check if there is a complex dielectric
    complexDiel = 0 
    if type(surf_array[0].Kext[0][0,0])==numpy.complex128:
        complexDiel = 1 

    if complexDiel==1:
        M = zeros((Neq,Neq), complex)
    else:
        M = zeros((Neq,Neq))
    
    M_sym = []
    for i in range(len(surf_array)):
        tar = surf_array[i]

        if complexDiel==1:
            tar.KextDiag = zeros(tar.N, complex)
            tar.KpextDiag = zeros(tar.N, complex)
            tar.VextDiag = zeros(tar.N, complex)
            tar.KintDiag = zeros(tar.N, complex)
            tar.KpintDiag = zeros(tar.N, complex)
            tar.VintDiag = zeros(tar.N, complex)
        
        else:
            tar.KextDiag = zeros(tar.N)
            tar.KpextDiag = zeros(tar.N)
            tar.VextDiag = zeros(tar.N)
            tar.KintDiag = zeros(tar.N)
            tar.KpintDiag = zeros(tar.N)
            tar.VintDiag = zeros(tar.N)

        if tar.surf_type=='dirichlet_surface' or tar.surf_type=='neumann_surface' or tar.surf_type=='neumann_surface_hyper':
            M_sym.append([[]])
        else:
            M_sym.append([[],[]])
       
        for j in range(len(surf_array)):
            src = surf_array[j]
            if src.surf_type=='dirichlet_surface' or src.surf_type=='neumann_surface' or src.surf_type=='neumann_surface_hyper':
                M_sym[i][0].append([''])    
                if tar.surf_type!='dirichlet_surface' and tar.surf_type!='neumann_surface' and tar.surf_type!='neumann_surface_hyper':
                    M_sym[i][1].append([''])    
            else:
                M_sym[i][0].append(['',''])    
                if tar.surf_type!='dirichlet_surface' and tar.surf_type!='neumann_surface' and tar.surf_type!='neumann_surface_hyper':
                    M_sym[i][1].append(['',''])
        


#   Indices in M_sym: M_sym[tar_surf][internal or external][src_surf][K or V]
#   If tar is dirichlet or neumann [internal or external] always [0]
#   If src is dirichlet or neumann [K or V] always [0]
    
#   Ordering of the matrix will be same as ordering in input file
#   putting internal equation first and then external one.

    for i in range(len(surf_array)):
        tar = surf_array[i]
        if tar.surf_type=='dirichlet_surface' or tar.surf_type=='neumann_surface' or tar.surf_type=='neumann_surface_hyper':
            for j in range(len(surf_array)):
                src = surf_array[j]
                if src.surf_type=='dirichlet_surface':
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] = tar.Vext[j][:,:]
                    M_sym[i][0][j][0] += tar.VextSym[j]
#                   Store diagonal for preconditioner
                    if i==j:
                        tar.VextDiag[:] = diagonal(tar.Vext[j][:,:])
                        
                elif src.surf_type=='neumann_surface':
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] = tar.Kext[j][:,:]
                    M_sym[i][0][j][0] += tar.KextSym[j]
#                   Store diagonal for preconditioner
                    if i==j:
                        tar.KextDiag[:] = diagonal(tar.Kext[j][:,:])
                     
                elif src.surf_type=='neumann_surface_hyper':
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] = tar.Kpext[j][:,:]
                    M_sym[i][0][j][0] += tar.KpextSym[j]
#                   Store diagonal for preconditioner
                    if i==j:
                        tar.KpextDiag[:] = diagonal(tar.Kpext[j][:,:])
                                    
                else:
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] = tar.Kext[j][:,:] 
                    M[tar.N0:tar.N0+tar.N,src.N0+src.N:src.N0+2*src.N] = tar.Vext[j][:,:]
                    M_sym[i][0][j][0] += tar.KextSym[j]
                    M_sym[i][0][j][1] += tar.VextSym[j]

        else:
            for j in range(len(surf_array)):
                src = surf_array[j]

                if src.surf_type=='dirichlet_surface':
#                   Internal equation
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] += tar.Vint[j][:,:]
                    M_sym[i][0][j][0] += tar.VintSym[j]
#                   External equation
                    M[tar.N0+tar.N:tar.N0+2*tar.N,src.N0:src.N0+src.N] += tar.Vext[j][:,:]
                    M_sym[i][1][j][0] += tar.VextSym[j]
                        
                elif src.surf_type=='neumann_surface':
#                   Internal equation
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] += tar.Kint[j][:,:]
                    M_sym[i][0][j][0] += tar.KintSym[j]
#                   External equation
                    M[tar.N0+tar.N:tar.N0+2*tar.N,src.N0:src.N0+src.N] += tar.Kext[j][:,:]
                    M_sym[i][1][j][0] += tar.KextSym[j]
                        
                elif src.surf_type=='neumann_surface_hyper':
#                   Internal equation
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] += tar.Kpint[j][:,:]
                    M_sym[i][0][j][0] += tar.KpintSym[j]
#                   External equation
                    M[tar.N0+tar.N:tar.N0+2*tar.N,src.N0:src.N0+src.N] += tar.Kpext[j][:,:]
                    M_sym[i][1][j][0] += tar.KpextSym[j]

                else:
#                   Internal equation
                    M[tar.N0:tar.N0+tar.N,src.N0:src.N0+src.N] += tar.Kint[j][:,:] 
                    M[tar.N0:tar.N0+tar.N,src.N0+src.N:src.N0+2*src.N] += tar.Vint[j][:,:]
                    M_sym[i][0][j][0] += tar.KintSym[j]
                    M_sym[i][0][j][1] += tar.VintSym[j]

#                   External equation
                    M[tar.N0+tar.N:tar.N0+2*tar.N,src.N0:src.N0+src.N] += tar.Kext[j][:,:] 
                    M[tar.N0+tar.N:tar.N0+2*tar.N,src.N0+src.N:src.N0+2*src.N] += tar.Vext[j][:,:]
                    M_sym[i][1][j][0] += tar.KextSym[j]
                    M_sym[i][1][j][1] += tar.VextSym[j]

#                   Store diagonal for preconditioner
                    if i==j:
                        tar.KextDiag[:] = diagonal(tar.Kext[j][:,:])
                        tar.VextDiag[:] = diagonal(tar.Vext[j][:,:])
                        tar.KintDiag[:] = diagonal(tar.Kint[j][:,:])
                        tar.VintDiag[:] = diagonal(tar.Vint[j][:,:])

    for i in range(len(M_sym)):
        for j in range(len(M_sym[i])):
            for k in range(len(M_sym[i][j])):
                for l in range(len(M_sym[i][j][k])):
                    if len(M_sym[i][j][k][l])==0:
                        M_sym[i][j][k][l] += '          0'


    return M, M_sym
        
def generatePreconditioner(surf_array):

    data_inv = []
    row = []
    col = []
    for s in surf_array:
        if s.surf_type=='dirichlet_surface':
#            Ainv11 = ones(len(s.VextDiag))
            Ainv11 = 1/s.VextDiag
            data_inv.extend(Ainv11)
            row.extend(range(s.N0,s.N0+s.N))
            col.extend(range(s.N0,s.N0+s.N))
        elif s.surf_type=='neumann_surface':
#            Ainv11 = ones(len(s.KextDiag))
            Ainv11 = 1/s.KextDiag
            data_inv.extend(Ainv11)
            row.extend(range(s.N0,s.N0+s.N))
            col.extend(range(s.N0,s.N0+s.N))
        elif s.surf_type=='neumann_surface_hyper':
#            Ainv11 = ones(len(s.KpextDiag))
            Ainv11 = 1/s.KpextDiag
            data_inv.extend(Ainv11)
            row.extend(range(s.N0,s.N0+s.N))
            col.extend(range(s.N0,s.N0+s.N))        
        else:
            dX11 = s.KintDiag
            dX12 = s.VintDiag
            dX21 = s.KextDiag
            dX22 = s.VextDiag
            d_aux = 1/(dX22-dX21*dX12/dX11)
            Ainv11 = 1/dX11 + 1/dX11*dX12*d_aux*dX21/dX11
            Ainv12 = -1/dX11*dX12*d_aux
            Ainv21 = -d_aux*dX21/dX11
            Ainv22 = d_aux
#           Top-left block
            data_inv.extend(Ainv11)
            row.extend(range(s.N0,s.N0+s.N))
            col.extend(range(s.N0,s.N0+s.N))
#           Top-right block
            data_inv.extend(Ainv12)
            row.extend(range(s.N0,s.N0+s.N))
            col.extend(range(s.N0+s.N,s.N0+2*s.N))
#           Bottom-left block
            data_inv.extend(Ainv21)
            row.extend(range(s.N0+s.N,s.N0+2*s.N))
            col.extend(range(s.N0,s.N0+s.N))
#           Bottom-right block
            data_inv.extend(Ainv22)
            row.extend(range(s.N0+s.N,s.N0+2*s.N))
            col.extend(range(s.N0+s.N,s.N0+2*s.N))
            
    data_inv = array(data_inv)
    col = array(col)
    row = array(row)
    Ainv = coo_matrix((data_inv,(row,col)))

    return Ainv
