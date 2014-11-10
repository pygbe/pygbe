#!/usr/bin/env python
from numpy import *

import sys
import os
sys.path.append('../util')
from readData import readTriangle, readVertex

def zeroAreas(vertex, triangle_raw, Area_null):
    for i in range(len(triangle_raw)):
        L0 = vertex[triangle_raw[i,1]] - vertex[triangle_raw[i,0]]
        L2 = vertex[triangle_raw[i,0]] - vertex[triangle_raw[i,2]]
        normal_aux = cross(L0,L2)
        Area_aux = linalg.norm(normal_aux)/2
        if Area_aux<1e-10:
            Area_null.append(i)
    return Area_null 

## Designed for a cube which faces are aligned with cartesian coordinates
meshFile = sys.argv[1]
x_right = float(sys.argv[2])
x_left  = float(sys.argv[3])
y_top   = float(sys.argv[4])
y_bott  = float(sys.argv[5])
z_front = float(sys.argv[6])
z_back  = float(sys.argv[7])

vertex = readVertex(meshFile+'.vert', float)
triangle_raw = readTriangle(meshFile+'.face', 'neumann_surface')

Area_null = []
Area_null = zeroAreas(vertex, triangle_raw, Area_null)
triangle = delete(triangle_raw, Area_null, 0)

if len(triangle) != len(triangle_raw):
    print '%i deleted triangles'%(len(triangle_raw)-len(triangle))

phi0 = zeros(len(triangle), float)

tri_ctr = average(vertex[triangle], axis=1)
print len(tri_ctr)
print len(triangle)

max_x = max(tri_ctr[:,0])
min_x = min(tri_ctr[:,0])
max_y = max(tri_ctr[:,1])
min_y = min(tri_ctr[:,1])
max_z = max(tri_ctr[:,2])
min_z = min(tri_ctr[:,2])

for i in range(len(triangle)):
    if abs(tri_ctr[i,0]-max_x)<1e-10:
        phi0[i] = x_right
    if abs(tri_ctr[i,0]-min_x)<1e-10:
        phi0[i] = x_left 
    if abs(tri_ctr[i,1]-max_y)<1e-10:
        phi0[i] = y_top  
    if abs(tri_ctr[i,1]-min_y)<1e-10:
        phi0[i] = y_bott 
    if abs(tri_ctr[i,2]-max_z)<1e-10:
        phi0[i] = z_front
    if abs(tri_ctr[i,2]-min_z)<1e-10:
        phi0[i] = z_back 

file_out = meshFile+'_'+sys.argv[2]+sys.argv[3]+sys.argv[4]+sys.argv[5]+sys.argv[6]+sys.argv[7]+'.phi0'
savetxt(file_out, phi0)
os.system('mv '+file_out+' input_files/')
