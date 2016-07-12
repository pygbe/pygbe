"""
It generates the mesh files (.vert and .face) for a brick.
"""

import numpy

def meshSurf(C, N, S, fix, normal):
    """
    It generates a triangular mesh for a rectangular surface.

    Arguments:
    ----------
    C     : (6, 3) list, Face centers.
    N     : (6, 2) list, Face nodes.
    S     : (6, 2) list, Face size, length of the face.
    fix   : (6, 1) list, contains the direction where the normal of the faces
                         lies on. Elements can be 'x', 'y' or 'z'.
    normal: (6, 1) list, normal direction. Elements can be 'pos' or 'neg'.

    Returns:
    --------
    nodes    : list, triangles vertices.
    triangles: list, indices of the triangles.
    """
    # xi, yi local 2D coordinates of the surface

    h = numpy.zeros(2, dtype=float)
    h[0] = S[0]/(N[0]-1) # Mesh size xi direction
    h[1] = S[1]/(N[1]-1) # Mesh size yi direction 

    if fix=='x':
        x0 = C[1] - S[0]/2   # Starting point xi direction
        y0 = C[2] - S[1]/2   # Starting point yi direction

    if fix=='y':
        x0 = C[0] - S[0]/2   # Starting point xi direction
        y0 = C[2] - S[1]/2   # Starting point yi direction

    if fix=='z':
        x0 = C[0] - S[0]/2   # Starting point xi direction
        y0 = C[1] - S[1]/2   # Starting point yi direction

    # Generate nodes
    xi,yi = numpy.mgrid[x0:x0+S[0]+h[0]/2.:h[0],y0:y0+S[1]+h[1]/2.:h[1]]
    xi = xi.flatten()
    yi = yi.flatten()
    
    nodes = numpy.zeros((N[0]*N[1], 3), dtype=numpy.float64)
    for ii in range(N[0]*N[1]):
        if fix=='x':
            nodes[ii,0] = C[0]
            nodes[ii,1] = xi[ii]
            nodes[ii,2] = yi[ii]
        if fix=='y':
            nodes[ii,0] = xi[ii]
            nodes[ii,1] = C[1]
            nodes[ii,2] = yi[ii]
        if fix=='z':
            nodes[ii,0] = xi[ii]
            nodes[ii,1] = yi[ii]
            nodes[ii,2] = C[2]

    # Generate triangles (RHS rule normal into page)
    Nt = 2*(N[0]-1)*(N[1]-1) # Number of triangles
    triangles = numpy.zeros((Nt,3), dtype=int)

    # Triangles pointing up
    counter = -1
    for ii in range(N[0]-1):
        for jj in range(N[1]-1):
            counter += 1
            V1 = ii*N[1] + jj
            V2 = ii*N[1] + jj+1
            V3 = (ii+1)*N[1] + jj
            if normal=='pos':
                triangles[counter,0] = V1
                triangles[counter,1] = V2
                triangles[counter,2] = V3
            if normal=='neg':
                triangles[counter,0] = V1
                triangles[counter,1] = V3
                triangles[counter,2] = V2

    # Triangles pointing down
    for ii in range(1,N[0]):
        for jj in range(1,N[1]):
            counter += 1
            V1 = ii*N[1] + jj
            V2 = ii*N[1] + jj-1
            V3 = (ii-1)*N[1] + jj
            if normal=='pos':
                triangles[counter,0] = V1
                triangles[counter,1] = V2
                triangles[counter,2] = V3
            if normal=='neg':
                triangles[counter,0] = V1
                triangles[counter,1] = V3
                triangles[counter,2] = V2

    return nodes, triangles


#     y | 
#       |____ x
#      / 
#   z / 
# Cube info
sz  = [250., 10., 250.] # Cube size
ctr = [0.,-sz[1]/2,0.]  # Cube center
d = 8                   # Density :triangles per angstrom square. 
sqr_per_side = int(numpy.ceil(numpy.sqrt(sz[0]*sz[2]*d/2.)))
avg_size = sz[0]/sqr_per_side
sqr_per_side_y = int(sz[1]/avg_size)
nod = [sqr_per_side+1, sqr_per_side_y+1, sqr_per_side+1]    # Number of nodes
print avg_size, sqr_per_side, sqr_per_side_y, nod

# Faces
# Face centers
C = [ [ctr[0]+sz[0]/2, ctr[1], ctr[2]],     # Right
      [ctr[0]-sz[0]/2, ctr[1], ctr[2]],     # Left
      [ctr[0], ctr[1]+sz[1]/2, ctr[2]],     # Up
      [ctr[0], ctr[1]-sz[1]/2, ctr[2]],     # Down
      [ctr[0], ctr[1], ctr[2]+sz[2]/2],     # Front 
      [ctr[0], ctr[1], ctr[2]-sz[2]/2] ]    # Back

# Face nodes
N = [ [nod[1], nod[2]],     # Right
      [nod[1], nod[2]],     # Left
      [nod[0], nod[2]],     # Up
      [nod[0], nod[2]],     # Down
      [nod[0], nod[1]],     # Front
      [nod[0], nod[1]] ]    # Back

# Face size
S = [ [sz[1], sz[2]],     # Right
      [sz[1], sz[2]],     # Left
      [sz[0], sz[2]],     # Up
      [sz[0], sz[2]],     # Down
      [sz[0], sz[1]],     # Front
      [sz[0], sz[1]] ]    # Back

# Face fix
fix = ['x', 'x', 'y', 'y', 'z', 'z']

# Normal direction
normal = ['neg','pos','pos','neg','neg','pos']

# Loop over faces
triangles = []
nodes = []
for i in range(6):
    node_face, triangle_face = meshSurf(C[i], N[i], S[i], fix[i], normal[i])

    triangle_face += len(nodes)
    nodes.extend(node_face)
    triangles.extend(triangle_face)

triangles = numpy.array(triangles) + 1 # Add 1 to conform to msms format
nodes = numpy.array(nodes)

print 'Cube center: %f, %f, %f'%(ctr[0],ctr[1],ctr[2])
print 'Cube size  : %f, %f, %f'%(sz[0],sz[1],sz[2])
print numpy.shape(nodes)
print numpy.shape(triangles)

if d>=10:
    mesh_out = 'sensor_%ix%ix%i_d%i'%(int(sz[0]), int(sz[1]), int(sz[2]), int(d))
else:
    mesh_out = 'sensor_%ix%ix%i_d0%i'%(int(sz[0]), int(sz[1]), int(sz[2]), int(d))
print 'Written to '+mesh_out
numpy.savetxt(mesh_out+'.face', triangles, fmt='%i')
numpy.savetxt(mesh_out+'.vert', nodes)

