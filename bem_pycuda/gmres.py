'''
    Copyright (C) 2011 by Christopher Cooper, Lorena Barba
  
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
from numpy  import zeros, array, dot, arange, exp, sqrt, random, transpose, sum
from numpy.linalg           import norm
from scipy.linalg           import lu_solve, solve
from scipy.sparse.linalg    import gmres 
import time
from matrixfree import gmres_dot as gmres_dot

def GeneratePlaneRotation(dx, dy, cs, sn):

    if dy==0:
        cs = 1. 
        sn = 0. 
    elif (abs(dy)>abs(dx)):
        temp = dx/dy
        sn = 1/sqrt(1+temp*temp)
        cs = temp*sn
    else:
        temp = dy/dx
        cs = 1/sqrt(1+temp*temp)
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

def gmres_solver (Precond, E_hat, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, 
					normal_xDev, xj, yj, zj, xi, yi, zi, xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                    xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, sizeTarDev, offsetMltDev, 
					Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                    tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost,
                    offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, K, threshold, BSZ, GSZ, BlocksPerTwig, X, b, 
                    R, tol, max_iter, Cells, theta, Nm, II, JJ, KK, index, combII, combJJ, combKK, 
                    IImii, JJmjj, KKmkk, index_small, P, kappa, NCRIT, twig, eps, time_pack):
    # Precond       	: preconditioner
    # E_hat         	: coefficient of bottom right block
    # vertex        	: array of position of vertices
    # triangle      	: array of indices of triangle vertices in vertex array
	# triHost/Dev		: packed array of triangle indices corresponding to each Gauss point on host/device memory	
	# kHost/Dev			: packed array of Gauss points indices within the element corresponding to each Gauss point on host/device memory	
	# vertexHost/Dev	: packed array of vertices of triangles on host/device memory
	# AreaHost/Dev		: packed array of area of triangles on host/device memory
	# normal_xDev		: packed array of x component of normal vector to triangles on device memory
    # xi, yi, zi    	: position of targets
    # xj, yj, zj    	: position of sources
	# xt,yt,ztHost/Dev	: packed array with position of targets (collocation points) on host/device memory
	# xs,ys,zsHost/Dev	: packed array with position of sources (Gauss points) on host/device memory
	# xc,yc,zcHost/Dev	: packed array with position of box centers on host/device memory
	# sizeTarHost/Dev	: array with number of targets per twig cell on host/device memory
	# offsetMltHost/Dev	: array with pointers to first element of each twig in xcHost/Dev array on host/device memory
	# Pre0,1,2,3Host/Dev: packed array with diagonal values of preconditioner for blocks 0,1,2,3 on host/device
	# tarPtr			: packed array with pointers to targets in xi, yi, zi array
	# srcPtr			: packed array with pointers to sources in xj, yj, zj array
	# offSrc			: length of array of packed sources
	# mltPtr			: packed array with pointers to multipoles in Cells array
	# offMlt			: length of array of packed multipoles
	# offsetSrcHost/Dev : array with pointers to first element of each twig in xsHost/Dev array on host/device memory
    # Area          	: array of element area 
    # normal        	: array of elements normal
    # xk/Dev, wk/Dev	: position and weight of 1D gauss quadrature on host/device memory
    # K             	: number of 2D Gauss points per element
	# threshold			: threshold to change from analytical to Gauss integration
	# BSZ, GSZ			: block size and grid size for CUDA
	# blocksPerTwig		: number of CUDA blocks that fit on a twig (NCRIT/BSZ)
    # X             	: solution vector
    # b             	: RHS vector
    # R             	: number of iterations to restart
    # tol           	: GMRES tolerance
    # max_iter      	: maximum number of iterations
    # Cells         	: array of Cells
    # theta         	: MAC criterion
    # Nm            	: number of terms in Taylor expansion
    # II, JJ, KK    	: x,y,z powers of multipole expansion
    # index         	: 1D mapping of II,JJ,KK (index of multipoles)
    # P             	: order of expansion
    # kappa         	: reciprocal of Debye length
    # NCRIT         	: max number of points per twig cell
    # twig          	: array of indices of twigs in Cells array

    N = len(b)
    V = zeros((R+1, N))
    H = zeros((R+1,R))

    time_Vi 		= 0.
    time_Vk 		= 0.
    time_rotation 	= 0.
    time_lu 		= 0.
    time_update 	= 0.

    ### Initialize varibles
    rel_resid = 1.
    cs, sn = zeros(N), zeros(N)

    iteration = 0

    b_norm = norm(b)

    time_eval = 0.
    time_an    = 0.
    time_P2P   = 0.
    time_P2M   = 0.
    time_M2M   = 0.
    time_M2P   = 0.

	### Outer loop
    while (iteration < max_iter and rel_resid>=tol): 
        
		# Call Treecode
        aux, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack, AI_int = gmres_dot(Precond, 
                    E_hat, X, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, 
					normal_xDev, xj, yj, zj, xi, yi, zi, xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                    xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, sizeTarDev, offsetMltDev, 
					Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
      				tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost, 
                    offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, Cells, theta, Nm, II, JJ, KK, 
                    index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, 
                    P, kappa, NCRIT, K, threshold, BSZ, GSZ, BlocksPerTwig, twig, eps, time_eval, time_P2P, 
                    time_P2M, time_M2M, time_M2P, time_an, time_pack)

       	# Check residual 
        r = b - aux
        beta = norm(r)

        if iteration==0: 
            print 'Analytical integrals: %i of %i, %i'%(AI_int*2/len(X), len(X)/2, 100*AI_int/len(X)**2)+'%'

        V[0,:] = r[:]/beta
        if iteration==0:
            res_0 = beta/b_norm

        s = zeros(R+1)
        s[0] = beta
        i = -1

        while (i+1<R and iteration+1<=max_iter): # Inner iteration
            i+=1 
            iteration+=1

            if iteration%10==0:
                print 'iteration: %i, residual %s'%(iteration,rel_resid)
                

            # Compute Vip1
            tic = time.time()
       
            Vip1, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack, AI_int = gmres_dot(Precond, 
                        E_hat, V[i,:], vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, normal_xDev, xj, yj, zj, xi, yi, zi, 
                        xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                        xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, 
                        sizeTarDev, offsetMltDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                        tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost, 
                        offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, Cells, theta, Nm, II, JJ, KK, 
                        index, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, 
                        P, kappa, NCRIT, K, threshold, BSZ, GSZ, BlocksPerTwig, twig, eps, time_eval, time_P2P, 
                        time_P2M, time_M2M, time_M2P, time_an, time_pack)
      

            toc = time.time()
            time_Vi+=toc-tic

            tic = time.time()
            Vk = V[0:i+1,:]
            H[0:i+1,i] = dot(Vip1,transpose(Vk))

            # This ends up being slower than looping           
#            HVk = H[0:i+1,i]*transpose(Vk)
#            Vip1 -= HVk.sum(axis=1)

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

            if (i+1==R):
                print('Residual: %f. Restart...'%rel_resid)
            if rel_resid<=tol:
                break

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
    print 'Time P2M          : %f'%time_P2M
    print 'Time M2M          : %f'%time_M2M
    print 'Time packing      : %f'%time_pack
    print 'Time evaluation   : %f'%time_eval
    print '\tTime M2P  : %f'%time_M2P
    print '\tTime P2P  : %f'%time_P2P
    print '\tTime analy: %f'%time_an
#    print 'Tolerance: %f, maximum iterations: %f'%(tol, max_iter)

    return X

"""
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
"""
