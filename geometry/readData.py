from numpy import *
from numpy import float64 as REAL

def readVertex(filename):
    x = []
    y = []
    z = []
    for line in file(filename):
        line = line.split()
        x0 = line[0]
        y0 = line[1]
        z0 = line[2]
        x.append(REAL(x0))
        y.append(REAL(y0))
        z.append(REAL(z0))

    x = array(x)
    y = array(y)
    z = array(z)
    return x, y, z

def readTriangle(filename):
    triangle = []

    for line in file(filename):
        line = line.split()
        v1 = line[0]
        v2 = line[2] # v2 and v3 are flipped to match my sign convention!
        v3 = line[1]
        triangle.append([int(v1)-1,int(v2)-1,int(v3)-1])
        # -1-> python starts from 0, matlab from 1

    triangle = array(triangle)

    return triangle

def readpqr(filename):

    pos = []
    q   = []

    start = 0
    for line in file(filename):
        line = array(line.split())
   
        if len(line)>8:# and start==2:
            x = line[4]
            y = line[5]
            z = line[6]
            q.append(REAL(line[9]))
            pos.append([REAL(x),REAL(y),REAL(z)])
    
        '''
        if len(line)==1:
            start += 1
            if start==2:
                Nq = int(line[0])
        '''
    pos = array(pos)
    q   = array(q)
    Nq  = len(q)
    return pos, q, Nq
