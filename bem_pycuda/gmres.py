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

import numpy
from scipy import linalg
import time
from matrixfree import gmres_dot as gmres_dot

def GeneratePlaneRotation(dx, dy, cs, sn):

    if dy==0:
        cs = 1. 
        sn = 0. 
    elif (abs(dy)>abs(dx)):
        temp = dx/dy
        sn = 1/numpy.sqrt(1+temp*temp)
        cs = temp*sn
    else:
        temp = dy/dx
        cs = 1/numpy.sqrt(1+temp*temp)
        sn = temp*cs

    return cs, sn

def ApplyPlaneRotation(dx, dy, cs, sn):
    temp = cs*dx + sn*dy
    dy = -sn*dx + cs*dy
    dx = temp

    return dx, dy

def PlaneRotation (H, cs, sn, s, i, R):
    for k in range(i):
        H[k,i], H[k+1,i] = ApplyPlaneRotation(H[k,i], H[k+1,i], cs[k], sn[k])
    
    cs[i],sn[i] = GeneratePlaneRotation(H[i,i], H[i+1,i], cs[i], sn[i]) 
    H[i,i],H[i+1,i] = ApplyPlaneRotation(H[i,i], H[i+1,i], cs[i], sn[i])
    s[i],s[i+1] = ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i])

    return H, cs, sn, s

def gmres_solver (surf_array, field_array, X, b, param, ind0, timing, kernel):

    N = len(b)
    V = numpy.zeros((param.restart+1, N))
    H = numpy.zeros((param.restart+1,param.restart))

    time_Vi = 0.
    time_Vk = 0.
    time_rotation = 0.
    time_lu = 0.
    time_update = 0.

    # Initializing varibles
    rel_resid = 1.
    cs, sn = numpy.zeros(N), numpy.zeros(N)

    iteration = 0

    b_norm = linalg.norm(b)

    while (iteration < param.max_iter and rel_resid>=param.tol): # Outer iteration
        
        aux = gmres_dot(X, surf_array, field_array, ind0, param, timing, kernel)
        
        r = b - aux
        beta = linalg.norm(r)

        if iteration==0: 
            print 'Analytical integrals: %i of %i, %i'%(timing.AI_int/param.N, param.N, 100*timing.AI_int/param.N**2)+'%'

        V[0,:] = r[:]/beta
        if iteration==0:
            res_0 = b_norm

        s = numpy.zeros(param.restart+1)
        s[0] = beta
        i = -1

        while (i+1<param.restart and iteration+1<=param.max_iter): # Inner iteration
            i+=1 
            iteration+=1

            # Compute Vip1
            tic = time.time()
       
            Vip1 = gmres_dot(V[i,:], surf_array, field_array, ind0, param, timing, kernel)
            toc = time.time()
            time_Vi+=toc-tic
    
            if iteration<6:
                numpy.savetxt('Vip1%i.txt'%iteration, Vip1)

            tic = time.time()
            Vk = V[0:i+1,:]
            H[0:i+1,i] = numpy.dot(Vip1,numpy.transpose(Vk))

            # This ends up being slower than looping           
#            HVk = H[0:i+1,i]*numpy.transpose(Vk)
#            Vip1 -= HVk.numpy.sum(axis=1)

            for k in range(i+1):
                Vip1 -= H[k,i]*Vk[k] 
            toc = time.time()
            time_Vk+=toc-tic

            H[i+1,i] = linalg.norm(Vip1)
            V[i+1,:] = Vip1[:]/H[i+1,i]

            tic = time.time()
            H,cs,sn,s =  PlaneRotation(H, cs, sn, s, i, param.restart)
            toc = time.time()
            time_rotation+=toc-tic

            rel_resid = abs(s[i+1])/res_0

            if iteration%1==0:
                print 'iteration: %i, rel resid: %s'%(iteration,rel_resid)


            if (i+1==param.restart):
                print('Residual: %f. Restart...'%rel_resid)
            if rel_resid<=param.tol:
                break

        # Solve the triangular system
        tic = time.time()
        piv = numpy.arange(i+1)
        y = linalg.lu_solve((H[0:i+1,0:i+1], piv), s[0:i+1], trans=0)
        toc = time.time()
        time_lu+=toc-tic

        # Update solution
        tic = time.time()
        Vj = numpy.zeros(N)
        for j in range(i+1):
            # Compute Vj
            Vj[:] = V[j,:]
            X += y[j]*Vj
        toc = time.time()
        time_update+=toc-tic


#    print 'Time Vip1    : %fs'%time_Vi
#    print 'Time Vk      : %fs'%time_Vk
#    print 'Time rotation: %fs'%time_rotation
#    print 'Time lu      : %fs'%time_lu
#    print 'Time update  : %fs'%time_update
    print 'GMRES solve'
    print 'Converged after %i iterations to a residual of %s'%(iteration,rel_resid)
    print 'Time weight vector: %f'%timing.time_mass
    print 'Time sort         : %f'%timing.time_sort
    print 'Time data transfer: %f'%timing.time_trans
    print 'Time P2M          : %f'%timing.time_P2M
    print 'Time M2M          : %f'%timing.time_M2M
    print 'Time M2P          : %f'%timing.time_M2P
    print 'Time P2P          : %f'%timing.time_P2P
    print '\tTime analy: %f'%timing.time_an
#    print 'Tolerance: %f, maximum iterations: %f'%(tol, max_iter)

    return X

"""
## Testing
from scipy.sparse.linalg  import gmres 

xmin = -1.
xmax = 1.
N = 5000
h = (xmax-xmin)/(N-1)
x = numpy.arange(xmin, xmax+h/2, h)

A = numpy.zeros((N,N))
for i in range(N):
    A[i] = numpy.exp(-abs(x-x[i])**2/(2*h**2))

b = numpy.random.random(N)
x = numpy.zeros(N)
R = 50
max_iter = 5000
tol = 1e-8

tic = time.time()
x = gmres_solver(A, x, b, R, tol, max_iter)
toc = time.time()
print 'Time for my GMRES: %fs'%(toc-tic)

tic = time.time()
xs = linalg.solve(A, b)
toc = time.time()
print 'Time for stright solve: %fs'%(toc-tic)


tic = time.time()
xg = gmres(A, b, x, tol, R, max_iter)[0]
toc = time.time()
print 'Time for scipy GMRES: %fs'%(toc-tic)


error = numpy.sqrt(numpy.sum((xs-x)**2)/numpy.sum(xs**2))
print 'error: %s'%error
"""
