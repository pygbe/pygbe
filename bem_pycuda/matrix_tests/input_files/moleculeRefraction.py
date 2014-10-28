"""
This code computes the refractive index of a protein, 
using PaliwalTomarGupta2014 Uricase as a base.
We use the relation:
n^2 = A + B/(1-C/L^2) + D/(1-E/L^2)
n: real part of refraction index
L: wavelength
A=-97.26
B=95.053
C=0.016
D=0.0647
E=0.521
We assume imaginary part of refractive index=0.024 (constant)
"""
import numpy

A=-97.26
B=95.053
C=0.016
D=0.0647
E=0.521

N=41

wl = numpy.linspace(0.515, 0.535, N)

n2 = A + B/(1-C/wl**2) + D/(1-E/wl**2)

n = numpy.sqrt(n2)

k = numpy.ones(N)*0.024

data = numpy.zeros((N,3))

data[:,0] = wl
data[:,1] = n
data[:,2] = k

numpy.savetxt('molecule_PTG14', data)
