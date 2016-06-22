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
from readData_old import (readFields, read_surface, readVertex, readTriangle,
                          readpqr)
from GaussIntegration import getGaussPoints


class surfaces():
    def __init__ (self):
        self.surf_type  = 0
        self.radius     = 0
        self.triangle   = []
        self.vertex     = []
        self.xi         = []
        self.yi         = []
        self.zi         = []
        self.xj         = []
        self.yj         = []
        self.zj         = []
        self.normal     = []
        self.Area       = []
        self.N0         = 0         # Start location in RHS
        self.N          = 0         # Number of panels
        self.phi0       = []
        self.Ein        = 0
        self.Eout       = 0
        self.Ehat       = 0
        self.kappa_in   = 0
        self.kappa_out  = 0
        self.Kint       = []        # Store K for internal equation
        self.Kpint      = []        # Store Kp for internal equation
        self.Vint       = []        # Store V for internal equation
        self.Kext       = []        # Store K for external equation 
        self.Kpext      = []        # Store Kp for external equation 
        self.Vext       = []        # Store V for external equation 
        self.KintSym    = []        # Store K for symbolic internal equation
        self.KpintSym   = []        # Store Kp for symbolic internal equation
        self.VintSym    = []        # Store V for symbolic internal equation
        self.KextSym    = []        # Store K for symbolic external equation 
        self.KpextSym   = []        # Store Kp for symbolic external equation 
        self.VextSym    = []        # Store V for symbolic external equation 
        self.KextDiag   = []        # Store diagonal of K external 
        self.KpextDiag  = []        # Store diagonal of Kp external 
        self.VextDiag   = []        # Store diagonal of V external 
        self.KintDiag   = []        # Store diagonal of K internal 
        self.KpintDiag  = []        # Store diagonal of Kp internal 
        self.VintDiag   = []        # Store diagonal of V internal 
        self.phi        = []        # Store value of potential on surface
        self.dphi       = []        # Store value of derivative of potential on surface
        self.dipole     = []        # Store dipole moment vector from surface

class parameters():
    def __init__(self):
        self.kappa         = 0.              # inverse of Debye length
        self.restart       = 0               # Restart of GMRES
        self.tol           = 0.              # Tolerance of GMRES
        self.max_iter      = 0               # Max number of GMRES iterations
        self.eps           = 0               # Epsilon machine
        self.K             = 0               # Number of Gauss points per element
        self.K_fine        = 0               # Number of Gauss points per element for near singular integrals
        self.Nk            = 0               # Gauss points per side for semi-analytical integrals
        self.threshold     = 0.              # L/d criterion for semi-analytic intergrals
        self.N             = 0               # Total number of elements
        self.Neq           = 0               # Total number of equations
        self.qe            = 1.60217646e-19  # Charge of an electron
        self.Na            = 6.0221415e23    # Avogadro's number
        self.E_0           = 8.854187818e-12 # Vacuum dielectric constant
        self.E_field       = []              # Regions where energy will be calculated

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


def readParameters(param, filename):

    val  = []
    for line in file(filename):
        line = line.split()
        val.append(line[1])

    param.K         = int (val[1])      # Gauss points per element
    param.Nk        = int (val[2])      # Number of Gauss points per side 
                                        # for semi analytical integral
    param.K_fine    = int (val[3])      # Number of Gauss points per element 
                                        # for near singular integrals 
    param.threshold = float(val[4])      # L/d threshold to use analytical integrals
                                        # Over: analytical, under: quadrature
    param.restart   = int (val[6])      # Restart for GMRES
    param.tol       = float(val[7])      # Tolerance for GMRES
    param.max_iter  = int (val[8])      # Max number of iteration for GMRES
    param.eps       = float(val[10])     # Epsilon machine
    param.theta     = float(val[12])     # MAC criterion for treecode

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
            field_aux.LorY  = int(LorY[i])                          # Laplace of Yukawa
        except ValueError:
            field_aux.LorY  = 0 
                                                                    # Dielectric constant
        if 'j' in E[i]:                                             # If dielectric constant is complex
            field_aux.E = complex(E[i])
        else:
            try:
                field_aux.E  = float(E[i])                          
            except ValueError:
                field_aux.E  = 0 
        try:
            field_aux.kappa = float(kappa[i])                      # inverse Debye length
        except ValueError:
            field_aux.kappa = 0 

        field_aux.coulomb = int(coulomb[i])                         # do/don't coulomb interaction
        if int(charges[i])==1:                                      # if there are charges
            if qfile[i][-4:]=='.crd':
                xq,q,Nq = readcrd(qfile[i], float)                  # read charges
            if qfile[i][-4:]=='.pqr':
                xq,q,Nq = readpqr(qfile[i], float)                  # read charges
            field_aux.xq = xq                                       # charges positions
            field_aux.q = q                                         # charges values
        if int(Nparent[i])==1:                                      # if it is an enclosed region
            field_aux.parent.append(int(parent[i]))                 # pointer to parent surface (enclosing surface)
        if int(Nchild[i])>0:                                        # if there are enclosed regions inside
            for j in range(int(Nchild[i])):
                field_aux.child.append(int(child[Nchild_aux+j]))    # Loop over children to get pointers
            Nchild_aux += int(Nchild[i])-1                          # Point to child for next surface
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


def initializeSurf(field_array, param, filename):

    surf_array = []

    files, surf_type, phi0_file = read_surface(filename)      # Read filenames for surfaces
    Nsurf = len(files)

    for i in range(Nsurf):
        print '\nReading surface %i from file '%i + files[i]

        s = surfaces()

        s.surf_type = surf_type[i]

        if s.surf_type=='dirichlet_surface' or s.surf_type=='neumann_surface' or s.surf_type=='neumann_surface_hyper':
            s.phi0 = loadtxt(phi0_file[i])

        Area_null = []
        s.vertex = readVertex(files[i]+'.vert', float)
        triangle_raw = readTriangle(files[i]+'.face', s.surf_type)
        Area_null = zeroAreas(s, triangle_raw, Area_null)
        s.triangle = delete(triangle_raw, Area_null, 0)
        print 'Removed areas=0: %i'%len(Area_null)

        # Look for regions inside/outside
        for j in range(Nsurf+1):
            if len(field_array[j].parent)>0:
                if field_array[j].parent[0]==i:                 # Inside region
                    s.kappa_in = field_array[j].kappa
                    s.Ein = field_array[j].E
            if len(field_array[j].child)>0:
                if i in field_array[j].child:                # Outside region
                    s.kappa_out = field_array[j].kappa
                    s.Eout = field_array[j].E

        if s.surf_type!='dirichlet_surface' and s.surf_type!='neumann_surface' and s.surf_type!='neumann_surface_hyper':
            s.Ehat = s.Ein/s.Eout
        else:
            s.Ehat = 1

        s.xi = average(s.vertex[s.triangle[:],0], axis=1)
        s.yi = average(s.vertex[s.triangle[:],1], axis=1)
        s.zi = average(s.vertex[s.triangle[:],2], axis=1)

        N = len(s.xi)

        s.normal = zeros((N,3))
        s.Area = zeros(N)
        for i in range(N):
            L0 = s.vertex[s.triangle[i,1]] - s.vertex[s.triangle[i,0]]
            L2 = s.vertex[s.triangle[i,0]] - s.vertex[s.triangle[i,2]]
            s.normal[i,:] = cross(L0,L2)
            s.Area[i] = linalg.norm(s.normal[i,:])/2
            s.normal[i,:] = s.normal[i,:]/(2*s.Area[i])
        
        # Get Gauss points
        s.xj, s.yj, s.zj = getGaussPoints(s.vertex,s.triangle,param.K)

        surf_array.append(s)

#   Find location of equation
    i = -1
    Neq = 0 
    for s in surf_array:
        i += 1
        s.N0 = 0 
        for ss in range(i):
            if surf_array[ss].surf_type=='dirichlet_surface' or surf_array[ss].surf_type=='neumann_surface' or s.surf_type=='neumann_surface_hyper':
                s.N0 += len(surf_array[ss].xi)
            else:
                s.N0 += 2*len(surf_array[ss].xi)

        s.N = len(s.xi)

        if s.surf_type=='dirichlet_surface' or s.surf_type=='neumann_surface' or s.surf_type=='neumann_surface_hyper':
            Neq += s.N 
        else:
            Neq += 2*s.N

#   Check if there is a complex dielectric
    complexDiel = 0 
    for f in field_array:
        if type(f.E)==complex:
            complexDiel = 1 

#   Allocate K and V and prepare symbolic arrays
    for tar in surf_array:
        for src in surf_array:
            if complexDiel==1:
                tar.Kint.append(zeros((tar.N,src.N), complex))
                tar.Kpint.append(zeros((tar.N,src.N), complex))
                tar.Vint.append(zeros((tar.N,src.N), complex))
                tar.Kext.append(zeros((tar.N,src.N), complex))
                tar.Kpext.append(zeros((tar.N,src.N), complex))
                tar.Vext.append(zeros((tar.N,src.N), complex))
            else:
                tar.Kint.append(zeros((tar.N,src.N)))
                tar.Kpint.append(zeros((tar.N,src.N)))
                tar.Vint.append(zeros((tar.N,src.N)))
                tar.Kext.append(zeros((tar.N,src.N)))
                tar.Kpext.append(zeros((tar.N,src.N)))
                tar.Vext.append(zeros((tar.N,src.N)))
            tar.KintSym.append('') 
            tar.KpintSym.append('') 
            tar.VintSym.append('')
            tar.KextSym.append('')
            tar.KpextSym.append('')
            tar.VextSym.append('')
            
    return surf_array, Neq

def readElectricField(filename):

    electricField = 0
    wavelength = 0
    for line in file(filename):
        line = line.split()

        if len(line)>0:
            if line[0]=='WAVE':
                electricField = float(line[1])
                wavelength = float(line[2])

    return electricField, wavelength

