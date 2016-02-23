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

from numpy import zeros, array, dot, arange, exp, sqrt, random, transpose, ravel, conjugate, savetxt, save, load
from numpy.linalg import norm
from scipy.linalg import lu_solve, solve
from scipy.sparse.linalg import gmres 
import time

def GeneratePlaneRotation(dx, dy, cs, sn):

    if dy==0:
        cs = 1. 
        sn = 0. 
    elif (abs(dy)>abs(dx)):
        temp = dx/dy
        sn = 1./sqrt(1+temp*temp)
        cs = temp*sn
    else:
        temp = dy/dx
        cs = 1./sqrt(1+temp*temp)
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

def gmres_solver (A, x, b, R, tol, max_iter):

    N = len(b)
    V = zeros((R+1, N))
    H = zeros((R+1,R))

    time_Vi = 0.
    time_Vk = 0.
    time_rotation = 0.
    time_lu = 0.
    time_update = 0.

    # Initializing varibles
    rel_resid = 1.
    cs, sn = zeros(N), zeros(N)

    iteration = 0

    b_norm = norm(b)

    while (iteration < max_iter and rel_resid>=tol): # Outer iteration
        
        r = b - dot(A,x)
        beta = norm(r)
        
        V[0,:] = r[:]/beta
        if iteration==0:
#            res_0 = beta/b_norm
            res_0 = b_norm

        s = zeros(R+1)
        s[0] = beta
        i = -1

        while (i+1<R and iteration+1<=max_iter): # Inner iteration
            i+=1 
            iteration+=1

            # Compute Vi
#            Vi = V[i,:]

            # Compute Vip1
            tic = time.time()
            Vip1 = dot(A, V[i,:])

            toc = time.time()
            time_Vi+=toc-tic

            if iteration<6:
                savetxt('Vip1%i_matrix.txt'%iteration, Vip1)
             
            tic = time.time()
            Vk = V[0:i+1,:]
            H[0:i+1,i] = dot(Vip1,transpose(Vk))

            # This ends up being slower than looping           
            #HVk = H[0:i+1,i]*transpose(Vk)
            #Vip1 -= HVk.sum(axis=1)
            #Vip1 -= transpose(map(sum,zip(*transpose(HVk))))

            for k in range(i+1):
                Vip1 -= H[k,i]*Vk[k] 
            toc = time.time()
            time_Vk+=toc-tic

            H[i+1,i] = norm(Vip1)
            V[i+1,:] = Vip1[:]/H[i+1,i]

            tic = time.time()
            H,cs,sn,s =  PlaneRotation(H, cs, sn, s, i, R)
            toc = time.time()
            time_rotation+=toc-tic

            rel_resid = abs(s[i+1])/res_0

            if iteration%1==0:
                print 'Iteration: %i, residual: %s'%(iteration,rel_resid)

            if rel_resid<tol:
                break

#            if (i+1==R):
#                print('Residual: %f. Restart...'%rel_resid)

        # Solve the triangular system
        tic = time.time()
        piv = arange(i+1)
        y = lu_solve((H[0:i+1,0:i+1], piv), s[0:i+1], trans=0)
        toc = time.time()
        time_lu+=toc-tic

        # Update solution
        tic = time.time()
       
        Vj = zeros(N)
        for j in range(i+1):
            # Compute Vj
            Vj[:] = V[j,:]
            x += y[j]*Vj
        
        toc = time.time()
        time_update+=toc-tic

    print 'Converged after %i iterations to a residual of %s'%(iteration,rel_resid)
#    print 'Time Vip1    : %fs'%time_Vi
#    print 'Time Vk      : %fs'%time_Vk
#    print 'Time rotation: %fs'%time_rotation
#    print 'Time lu      : %fs'%time_lu
#    print 'Time update  : %fs'%time_update
#    print 'Tolerance: %f, maximum iterations: %f'%(tol, max_iter)

    return x

'''
## Testing
xmin = -1.
xmax = 1.
N = 5000
h = (xmax-xmin)/(N-1)
x = arange(xmin, xmax+h/2, h)

A = zeros((N,N))
for i in range(N):
    A[i] = exp(-abs(x-x[i])**2/(2*h**2))

b = random.random(N)
x = zeros(N)
R = 50
max_iter = 5000
tol = 1e-8

tic = time.time()
x = gmres_solver(A, x, b, R, tol, max_iter)
toc = time.time()
print 'Time for my GMRES: %fs'%(toc-tic)

tic = time.time()
xs = solve(A, b)
toc = time.time()
print 'Time for stright solve: %fs'%(toc-tic)


tic = time.time()
xg = gmres(A, b, x, tol, R, max_iter)[0]
toc = time.time()
print 'Time for scipy GMRES: %fs'%(toc-tic)


error = sqrt(sum((xs-x)**2)/sum(xs**2))
print 'error: %s'%error
'''
