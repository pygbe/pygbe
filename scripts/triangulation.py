"""
Create a unitsphere recursively by subdividing all triangles in an octahedron
recursivly.

A unitsphere has a radius of 1, which also means that all points in this sphere
have an absolute value of 1. Another feature of an unitsphere is that the
normals of this sphere are exactly the same as the vertices.

This recursive method will avoid the common problem of the polar singularity, 
produced by 2d parameterization methods.

If you wish a sphere with another radius than that of 1, simply multiply every
single  value in the vertex array with this new radius (although this will
break the "vertex array equal to normal array" property)

Taken from https://sites.google.com/site/dlampetest/python/triangulating-a-
sphere-recursively 

Thanks!
"""
import numpy


octahedron_vertices = numpy.array( [ 
    [ 1.0, 0.0, 0.0], # 0 
    [-1.0, 0.0, 0.0], # 1
    [ 0.0, 1.0, 0.0], # 2 
    [ 0.0,-1.0, 0.0], # 3
    [ 0.0, 0.0, 1.0], # 4 
    [ 0.0, 0.0,-1.0]  # 5                                
    ] )
octahedron_triangles = numpy.array( [ 
    [ 0, 4, 2 ],
    [ 2, 4, 1 ],
    [ 1, 4, 3 ],
    [ 3, 4, 0 ],
    [ 0, 2, 5 ],
    [ 2, 1, 5 ],
    [ 1, 3, 5 ],
    [ 3, 0, 5 ]] )

def normalize_v3(arr):
    """
    Normalize a numpy array of 3 component vectors shape=(n,3)

    Arguments:
    ----------  
    arr: (n,3) array, desired array to be normalized.
 
    Returns:
    --------
    arr: (n,3) array, normalized array
    """
    lens = numpy.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def divide_all( vertices, triangles ):    
    """
    Subdivide each triangle in the old approximation and normalize the new
    points thus generated to lie on the surface of the unit
    sphere.
    Each input triangle with vertices labeled [0,1,2] as shown
    below will be turned into four new triangles:

                Make new points
                     a = (0+2)/2
                     b = (0+1)/2
                     c = (1+2)/2
            1
           /\        Normalize a, b, c
          /  \
        b/____\ c    Construct new triangles
        /\    /\       t1 [0,b,a]
       /  \  /  \      t2 [b,1,c]
      /____\/____\     t3 [a,b,c]
     0      a     2    t4 [a,c,2]    

    Arguments:
    ----------
    vertices  : array, vertices of the triangles.
    triangles : array, indices corresponding to the triangles. 

    Returns:
    --------
    vert_new  : array, vertices of the new triangles.
    triang_new: array, indices corresponding to the new triangles. 
    

    """ 
    new_triangle_count = len( triangles ) * 4
     
    v0 = vertices[ triangles[:,0] ]
    v1 = vertices[ triangles[:,1] ]
    v2 = vertices[ triangles[:,2] ]
    a = ( v0+v2 ) * 0.5
    b = ( v0+v1 ) * 0.5
    c = ( v1+v2 ) * 0.5  
    normalize_v3( a )
    normalize_v3( b )
    normalize_v3( c )
    
    #Stack the triangles together.
    vert_new  = numpy.hstack( (v0,b,a,  b,v1,c,  a,b,c, a,c,v2) ).reshape((-1,3))
    triang_new = numpy.arange( len(vertices) ).reshape( (-1,3) )
    #Now our vertices are duplicated, and thus our triangle structure
    # are unnecesarry.    
    return vert_new, triang_new

def create_unit_sphere( recursion_level=2 ):
    """
    It creates a unit sphere based on a recursive division of a octahedron.

    Arguments:
    ----------
    recursion_level: int, times we want to divide the octahedron.

    Returns:
    --------
    vertex_array   : array, vertices of the sphere triangles.
    index_array    : array, indices of the triangles.
    center         : array, contains centers of each triangle.

    """
    vertex_array, index_array = octahedron_vertices, octahedron_triangles
    for i in range( recursion_level - 1 ):
        vertex_array, index_array  = divide_all(vertex_array, index_array)

    center = numpy.zeros((len(index_array), 3))
    for i in range(len(index_array)):
        triangle = numpy.array([vertex_array[index_array[i,0]], vertex_array[
            index_array[i,1]], vertex_array[index_array[i,2]]])
        center[i,:] = numpy.dot(numpy.transpose(triangle), 1/3.*numpy.ones(3))
        
    return vertex_array, index_array, center


def vertex_array_only_unit_sphere( recursion_level=2 ):
    """
    It computes only the vertex array of a unit sphere.

    Arguments:
    ----------
    recursion_level: int, times we want to divide the octahedron.

    Returns:
    --------
    vertex_array   : array, vertices of the sphere triangles.
    """

    vertex_array, index_array = create_unit_sphere(recursion_level)
    if recursion_level > 1:    
        return vertex_array.reshape( (-1) )
    else:
        return vertex_array[index_array].reshape( (-1) )

def surfaceVariables(vertex, triangle):
    """
    It computes the variables 'normal' and 'Area' of a surface.

    Arguments:
    ----------
    vertex   : array, vertices of the triangles. 
    triangles: array, indices of the triangles. 

    Returns:
    --------
    normal    : array, contains the vector normal to the surface of each
                       triangle. 
    Area      : array, contains the area of each triangle.    
    """

    N = len(triangle)
    normal = numpy.zeros((N,3))
    Area   = numpy.zeros(N)
    for i in range(N):
        y = vertex[triangle[i]]
        L = numpy.array([y[1]-y[0], y[2]-y[1], y[0]-y[2]])
        normal[i,:] = numpy.cross(L[0],L[2])
        norm_normal = numpy.linalg.norm(normal[i,:])
        Area[i]     = norm_normal/2.
        normal[i,:] /= norm_normal
    
    return normal, Area


