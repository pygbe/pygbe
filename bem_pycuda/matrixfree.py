from numpy import *
import sys
sys.path.append('tree')
from FMMutils import *
from projection import project, get_phir, get_phir_gpu
from classes import parameters, index_constant
import time
sys.path.append('../util')
from semi_analytical import GQ_1D
from direct import coulomb_direct

# PyCUDA libraries
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

def selfInterior(surf, s, LorY, param, ind0, timing, kernel):
#    print 'SELF INTERIOR, surface: %i'%s
    K_diag = 2*pi
    V_diag = 0
    K_lyr, V_lyr = project(surf.XinK, surf.XinV, LorY, surf, surf, 
                            K_diag, V_diag, s, param, ind0, timing, kernel)
    v = K_lyr - V_lyr
    return v

def selfExterior(surf, s, LorY, param, ind0, timing, kernel):
#    print 'SELF EXTERIOR, surface: %i, E_hat: %f'%(s, surf.E_hat)
    K_diag = -2*pi
    V_diag = 0.
    K_lyr, V_lyr = project(surf.XinK, surf.XinV, LorY, surf, surf, 
                            K_diag, V_diag, s, param, ind0, timing, kernel)
    v = -K_lyr + surf.E_hat*V_lyr
    return v, K_lyr, V_lyr

def nonselfExterior(surf, src, tar, LorY, param, ind0, timing, kernel):
#    print 'NONSELF EXTERIOR, source: %i, target: %i, E_hat: %f'%(src,tar, surf[src].E_hat)
    K_diag = 0
    V_diag = 0
    K_lyr, V_lyr = project(surf[src].XinK, surf[src].XinV, LorY, surf[src], surf[tar], 
                            K_diag, V_diag, src, param, ind0, timing, kernel)
    v = -K_lyr + surf[src].E_hat*V_lyr
    return v

def nonselfInterior(surf, src, tar, LorY, param, ind0, timing, kernel):
#    print 'NONSELF INTERIOR, source: %i, target: %i'%(src,tar)
    K_diag = 0
    V_diag = 0
    K_lyr, V_lyr = project(surf[src].XinK, surf[src].XinV, LorY, surf[src], surf[tar], 
                            K_diag, V_diag, src, param, ind0, timing, kernel)
    v = K_lyr - V_lyr
    return v


def gmres_dot (X, surf_array, field_array, ind0, param, timing, kernel):
    
    Nfield = len(field_array)
    Nsurf = len(surf_array)

#   Place weights on corresponding surfaces and allocate memory
    Naux = 0
    for i in range(Nsurf):
        N = len(surf_array[i].triangle)
        surf_array[i].XinK     = X[Naux:Naux+N]
        surf_array[i].XinV     = X[Naux+N:Naux+2*N]
        surf_array[i].Xout_int = zeros(N) 
        surf_array[i].Xout_ext = zeros(N)
        Naux += 2*N 

#   Loop over fields
    for F in range(Nfield):
        LorY = field_array[F].LorY
        param.kappa = field_array[F].kappa
#        print '\n---------------------'
#        print 'REGION %i, LorY: %i, kappa: %f'%(F,LorY,param.kappa)

#       if parent surface -> self interior operator
        if len(field_array[F].parent)>0:
            p = field_array[F].parent[0]
            v = selfInterior(surf_array[p], p, LorY, param, ind0, timing, kernel)
            surf_array[p].Xout_int += v
            
#       if child surface -> self exterior operator + sibling interaction
#       sibling interaction: non-self exterior saved on exterior vector
        if len(field_array[F].child)>0:
            C = field_array[F].child
            for c1 in C:
                v,t1,t2 = selfExterior(surf_array[c1], c1, LorY, param, ind0, timing, kernel)
                surf_array[c1].Xout_ext += v
                for c2 in C:
                    if c1!=c2:
                        v = nonselfExterior(surf_array, c2, c1, LorY, param, ind0, timing, kernel)
                        surf_array[c1].Xout_ext += v

#       if child and parent surface -> parent-child and child-parent interaction
#       parent->child: non-self interior saved on exterior vector 
#       child->parent: non-self exterior saved on interior vector
        if len(field_array[F].child)>0 and len(field_array[F].parent)>0:
            p = field_array[F].parent[0]
            C = field_array[F].child
            for c in C:
                v = nonselfExterior(surf_array, c, p, LorY, param, ind0, timing, kernel)
                surf_array[p].Xout_int += v
                v = nonselfInterior(surf_array, p, c, LorY, param, ind0, timing, kernel)
                surf_array[c].Xout_ext += v
 
#   Gather results into the result vector
    MV = zeros(len(X))
    Naux = 0
    for i in range(Nsurf):
        N = len(surf_array[i].triangle)
        MV[Naux:Naux+N]     = surf_array[i].Xout_int*surf_array[i].Precond[0,:] + surf_array[i].Xout_ext*surf_array[i].Precond[1,:] 
        MV[Naux+N:Naux+2*N] = surf_array[i].Xout_int*surf_array[i].Precond[2,:] + surf_array[i].Xout_ext*surf_array[i].Precond[3,:] 
        Naux += 2*N

    return MV

def generateRHS(field_array, surf_array, N):
    F = zeros(2*N)
    for j in range(len(field_array)):
        Nq = len(field_array[j].q)
        if Nq>0:
#           First look at CHILD surfaces
            for s in field_array[j].child:          # Loop over surfaces
#           Locate position of surface s in RHS
                s_start = 0
                for ss in range(s):
                    s_start += 2*len(surf_array[ss].xi)
                s_size = len(surf_array[s].xi)

                aux = zeros(len(surf_array[s].xi))
                for i in range(Nq):
                    dx_pq = field_array[j].xq[i,0] - surf_array[s].xi
                    dy_pq = field_array[j].xq[i,1] - surf_array[s].yi
                    dz_pq = field_array[j].xq[i,2] - surf_array[s].zi
                    R_pq = sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)

                    aux += field_array[j].q[i]/(field_array[j].E*R_pq)

#               For CHILD surfaces, q contributes to RHS in 
#               EXTERIOR equation (hence Precond[1,:] and [3,:])
    
#               No preconditioner
#                F[s_start:s_start+s_size] += aux
#               With preconditioner
                F[s_start:s_start+s_size] += aux*surf_array[s].Precond[1,:]
                F[s_start+s_size:s_start+2*s_size] += aux*surf_array[s].Precond[3,:]

#           Now look at PARENT surface
            if len(field_array[j].parent)>0:
#           Locate position of surface s in RHS
                s = field_array[j].parent[0]
                s_start = 0
                for ss in range(s):
                    s_start += 2*len(surf_array[ss].xi)
                s_size = len(surf_array[s].xi)

                aux = zeros(len(surf_array[s].xi))
                for i in range(Nq):
                    dx_pq = field_array[j].xq[i,0] - surf_array[s].xi
                    dy_pq = field_array[j].xq[i,1] - surf_array[s].yi
                    dz_pq = field_array[j].xq[i,2] - surf_array[s].zi
                    R_pq = sqrt(dx_pq*dx_pq + dy_pq*dy_pq + dz_pq*dz_pq)

                    aux += field_array[j].q[i]/(field_array[j].E*R_pq)

#               No preconditioner
#                F[s_start:s_start+s_size] += aux
#               With preconditioner
                F[s_start:s_start+s_size] += aux*surf_array[s].Precond[0,:]
                F[s_start+s_size:s_start+2*s_size] += aux*surf_array[s].Precond[2,:]
               
    return F

def generateRHS_gpu(field_array, surf_array, param, kernel):

    F = zeros(2*param.N)
    REAL = param.REAL
    computeRHS_gpu = kernel.get_function("compute_RHS")
    for j in range(len(field_array)):
        Nq = len(field_array[j].q)
        if Nq>0:
        
#           First for CHILD surfaces
            for s in field_array[j].child[:]:       # Loop over surfaces
#           Locate position of surface s in RHS
                s_start = 0
                for ss in range(s):
                    s_start += 2*len(surf_array[ss].xi)
                s_size = len(surf_array[s].xi)
                Nround = len(surf_array[s].twig)*param.NCRIT

                GSZ = int(ceil(float(Nround)/param.NCRIT)) # CUDA grid size
                F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                computeRHS_gpu(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, field_array[j].q_gpu,
                                surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, int32(Nq), 
                                REAL(field_array[j].E), int32(param.NCRIT), int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                aux = zeros(Nround)
                F_gpu.get(aux)

#               For CHILD surfaces, q contributes to RHS in 
#               EXTERIOR equation (hence Precond[1,:] and [3,:])
    
#               No preconditioner
#                F[s_start:s_start+s_size] += aux
#               With preconditioner
                F[s_start:s_start+s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[1,:]
                F[s_start+s_size:s_start+2*s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[3,:]

#           Now for PARENT surface
            if len(field_array[j].parent)>0:
                s = field_array[j].parent[0]

#           Locate position of surface s in RHS
                s_start = 0
                for ss in range(s):
                    s_start += 2*len(surf_array[ss].xi)
                s_size = len(surf_array[s].xi)
                Nround = len(surf_array[s].twig)*param.NCRIT

                GSZ = int(ceil(float(Nround)/param.NCRIT)) # CUDA grid size
                F_gpu = gpuarray.zeros(Nround, dtype=REAL)     
                computeRHS_gpu(F_gpu, field_array[j].xq_gpu, field_array[j].yq_gpu, field_array[j].zq_gpu, field_array[j].q_gpu,
                                surf_array[s].xiDev, surf_array[s].yiDev, surf_array[s].ziDev, surf_array[s].sizeTarDev, int32(Nq), 
                                REAL(field_array[j].E), int32(param.NCRIT), int32(param.BlocksPerTwig), block=(param.BSZ,1,1), grid=(GSZ,1)) 

                aux = zeros(Nround)
                F_gpu.get(aux)

#               For PARENT surface, q contributes to RHS in 
#               INTERIOR equation (hence Precond[0,:] and [2,:])
    
#               No preconditioner
#                F[s_start:s_start+s_size] += aux
#               With preconditioner
                F[s_start:s_start+s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[0,:]
                F[s_start+s_size:s_start+2*s_size] += aux[surf_array[s].unsort]*surf_array[s].Precond[2,:]
 
    return F

def calculateEsolv(phi, surf_array, field_array, param, kernel):

    REAL = param.REAL

    par_reac = parameters()
    par_reac = param
    par_reac.threshold = 0.05
    par_reac.P = 7
    par_reac.theta = 0.0
    par_reac.Nm= (par_reac.P+1)*(par_reac.P+2)*(par_reac.P+3)/6

    ind_reac = index_constant()
    computeIndices(par_reac.P, ind_reac)
    precomputeTerms(par_reac.P, ind_reac)

    par_reac.Nk = 13         # Number of Gauss points per side for semi-analytical integrals

    JtoCal = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(JtoCal*param.E_0)
    E_solv = 0.
    AI_int = 0
    Naux = 0
    phi_reac = zeros(len(field_array[param.E_field].q))

#   First look at CHILD surfaces
#   Need to account for normals pointing outwards
#   and E_hat coefficient (as region is outside and 
#   dphi_dn is defined inside)
    for i in field_array[param.E_field].child:
        s = surf_array[i]
        s.xk,s.wk = GQ_1D(par_reac.Nk)
        s.xk = REAL(s.xk)
        s.wk = REAL(s.wk)
        for C in range(len(s.tree)):
            s.tree[C].M  = zeros(par_reac.Nm)
            s.tree[C].Md = zeros(par_reac.Nm)

        Naux += len(s.triangle)

#       Locate surface "s" in the phi array
        Nst = 0
        for ii in range(i):
            Nst += 2*len(surf_array[ii].triangle)
        Nsz = len(s.triangle)
    
#       Coefficient to account for dphi_dn defined in
#       interior but calculation done in exterior
        C1 = s.E_hat

        if param.GPU==0:
            phi_aux, AI = get_phir(phi[Nst:Nst+Nsz], C1*phi[Nst+Nsz:Nst+2*Nsz], s, field_array[param.E_field].xq, s.tree, par_reac, ind_reac)
        elif param.GPU==1:
            phi_aux, AI = get_phir_gpu(phi[Nst:Nst+Nsz], C1*phi[Nst+Nsz:Nst+2*Nsz], s, field_array[param.E_field], par_reac, kernel)
        
        AI_int += AI
        phi_reac -= phi_aux # Minus sign to account for normal pointing out

#   Now look at PARENT surface
    if len(field_array[param.E_field].parent)>0:
        i = field_array[param.E_field].parent[0]
        s = surf_array[i]
        s.xk,s.wk = GQ_1D(par_reac.Nk)
        s.xk = REAL(s.xk)
        s.wk = REAL(s.wk)
        for C in range(len(s.tree)):
            s.tree[C].M  = zeros(par_reac.Nm)
            s.tree[C].Md = zeros(par_reac.Nm)

        Naux += len(s.triangle)

#       Locate surface "s" in the phi array
        Nst = 0
        for ii in range(i):
            Nst += 2*len(surf_array[ii].triangle)
        Nsz = len(s.triangle)
    
        if param.GPU==0:
            phi_aux, AI = get_phir(phi[Nst:Nst+Nsz], phi[Nst+Nsz:Nst+2*Nsz], s, field_array[param.E_field].xq, s.tree, par_reac, ind_reac)
        elif param.GPU==1:
            phi_aux, AI = get_phir_gpu(phi[Nst:Nst+Nsz], phi[Nst+Nsz:Nst+2*Nsz], s, field_array[param.E_field], par_reac, kernel)
        
        AI_int += AI
        phi_reac += phi_aux 


    E_solv += 0.5*C0*sum(field_array[param.E_field].q*phi_reac)

    print '%i of %i analytical integrals for phi_reac calculation'%(AI_int/len(field_array[param.E_field].xq),Naux)
    return E_solv      


def coulombEnergy(f, param):

    point_energy = zeros(len(f.q), param.REAL)
    coulomb_direct(f.xq[:,0], f.xq[:,1], f.xq[:,2], f.q, point_energy)

    JtoCal = 4.184
    C0 = param.qe**2*param.Na*1e-3*1e10/(JtoCal*param.E_0)

    Ecoul = sum(point_energy) * 0.5*C0/(4*pi*f.E)
    return Ecoul
