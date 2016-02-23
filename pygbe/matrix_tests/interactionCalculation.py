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
from blockMatrixGen import blockMatrix
import sys
sys.path.append('../util')
from semi_analytical import GQ_1D
from GaussIntegration import getWeights

def computeInter(surf_array, field_array, param):

    WK = getWeights(param.K)

    xk,wk = GQ_1D(param.Nk)

    for f in field_array:
    
#       Effect on child surfaces
        for i in f.child:
            tar = surf_array[i]
#           Of child surfaces
            for j in f.child:
                print 'Target: %i, Source: %i'%(i,j)
                src = surf_array[j]
                K_lyr,V_lyr,Kp_lyr = blockMatrix(tar, src, WK, f.kappa, param.threshold, f.LorY, xk, wk, param.K_fine, param.eps)
                if i==j:    # Self-external
                    Diag = 2*pi*identity(len(K_lyr))

                    if f.LorY==1: # if Laplace
                        tar.Kext[j][:,:] = Diag - K_lyr[:,:]
                        tar.Kpext[j][:,:]= -Kp_lyr[:,:]
                        tar.Vext[j][:,:] = src.Ehat*V_lyr[:,:]
                        tar.KextSym[j]  += ' (1/2-KL%i%i)'%(i,j) 
                        tar.KpextSym[j] += '     -KpL%i%i'%(i,j) 
                        tar.VextSym[j]  += '    Eh%iVL%i%i'%(j,i,j) 
                    else:                   # if Yukawa
                        tar.Kext[j][:,:] = Diag - K_lyr[:,:]
                        tar.Kpext[j][:,:]= -Kp_lyr[:,:]
                        tar.Vext[j][:,:] = src.Ehat*V_lyr[:,:]
                        tar.KextSym[j]  += ' (1/2-KY%i%i)'%(i,j) 
                        tar.KpextSym[j] += '     -KpY%i%i'%(i,j) 
                        tar.VextSym[j]  += '    Eh%iVY%i%i'%(j,i,j) 

                else:
                    if f.LorY==1:   # if Laplace
                        tar.Kext[j][:,:] = -K_lyr[:,:]
                        tar.Kpext[j][:,:]= -Kp_lyr[:,:]
                        tar.Vext[j][:,:] =  src.Ehat*V_lyr[:,:]
                        tar.KextSym[j]  += '      -KL%i%i'%(i,j) 
                        tar.KpextSym[j] += '     -KpL%i%i'%(i,j) 
                        tar.VextSym[j]  += '    Eh%iVL%i%i'%(j,i,j) 
                    else:           # if Yukawa
                        tar.Kext[j][:,:] = -K_lyr[:,:]
                        tar.Kpext[j][:,:]= -Kp_lyr[:,:]
                        tar.Vext[j][:,:] =  src.Ehat*V_lyr[:,:]
                        tar.KextSym[j]  += '      -KY%i%i'%(i,j) 
                        tar.KpextSym[j] += '     -KpY%i%i'%(i,j) 
                        tar.VextSym[j]  += '    Eh%iVY%i%i'%(j,i,j) 
                
#           Of parent surface
            if len(f.parent)>0:
                j = f.parent[0]
                print 'Target: %i, Source: %i'%(i,j)
                src = surf_array[j]
                K_lyr,V_lyr,Kp_lyr = blockMatrix(tar, src, WK, f.kappa, param.threshold, f.LorY, xk, wk, param.K_fine, param.eps)
                if f.LorY==1:   # if Laplace
                    tar.Kext[j][:,:] =  K_lyr[:]
                    tar.Kpext[j][:,:]=  Kp_lyr[:]
                    tar.Vext[j][:,:] = -V_lyr[:]
                    tar.KextSym[j] += '       KL%i%i'%(i,j) 
                    tar.KpextSym[j]+= '      KpL%i%i'%(i,j) 
                    tar.VextSym[j] += '      -VL%i%i'%(i,j) 
                else:
                    tar.Kext[j][:,:] =  K_lyr[:]
                    tar.Kpext[j][:,:]=  Kp_lyr[:]
                    tar.Vext[j][:,:] = -V_lyr[:]
                    tar.KextSym[j] += '       KY%i%i'%(i,j) 
                    tar.KpextSym[j]+= '      KpY%i%i'%(i,j) 
                    tar.VextSym[j] += '      -VY%i%i'%(i,j) 

#       Effect on parent surface
        if len(f.parent)>0:
            i = f.parent[0]
            tar = surf_array[i]
#           Of child surfaces
            for j in f.child:
                print 'Target: %i, Source: %i'%(i,j)
                src = surf_array[j]
                K_lyr,V_lyr,Kp_lyr = blockMatrix(tar, src, WK, f.kappa, param.threshold, f.LorY, xk, wk, param.K_fine, param.eps)
                if f.LorY==1:   # if Laplace
                    tar.Kint[j] = -K_lyr[:]
                    tar.Kpint[j]= -Kp_lyr[:]
                    tar.Vint[j] =  src.Ehat*V_lyr[:]
                    tar.KintSym[j] += '      -KL%i%i'%(i,j) 
                    tar.KpintSym[j]+= '     -KpL%i%i'%(i,j) 
                    tar.VintSym[j] += '    Eh%iVL%i%i'%(j,i,j) 
                else:
                    tar.Kint[j] = -K_lyr[:]
                    tar.Kpint[j]= -Kp_lyr[:]
                    tar.Vint[j] =  src.Ehat*V_lyr[:]
                    tar.KintSym[j] += '      -KY%i%i'%(i,j) 
                    tar.KpintSym[j]+= '     -KpY%i%i'%(i,j) 
                    tar.VintSym[j] += '    Eh%iVY%i%i'%(j,i,j) 

#           Of parent surface (self-internal)
            j = i
            print 'Target: %i, Source: %i'%(i,j)
            src = surf_array[j]
            K_lyr,V_lyr,Kp_lyr = blockMatrix(tar, src, WK, f.kappa, param.threshold, f.LorY, xk, wk, param.K_fine, param.eps)
            Diag = 2*pi*identity(len(K_lyr))

            if f.LorY==1:  # if Laplace
                tar.Kint[j][:,:] = Diag + K_lyr[:,:]
                tar.Kpint[j][:,:]= Kp_lyr[:,:]
                tar.Vint[j][:,:] = -V_lyr[:,:]
                tar.KintSym[j]  += ' (1/2+KL%i%i)'%(i,j) 
                tar.KpintSym[j] += '      KpL%i%i'%(i,j) 
                tar.VintSym[j]  += '      -VL%i%i'%(i,j) 
            else:           # if Yukawa
                tar.Kint[j][:,:] = Diag + K_lyr[:,:]
                tar.Kpint[j][:,:]= Kp_lyr[:,:]
                tar.Vint[j][:,:] = -V_lyr[:,:]
                tar.KintSym[j]  += ' (1/2+KY%i%i)'%(i,j) 
                tar.KpintSym[j] += '      KpY%i%i'%(i,j) 
                tar.VintSym[j]  += '      -VY%i%i'%(i,j) 

