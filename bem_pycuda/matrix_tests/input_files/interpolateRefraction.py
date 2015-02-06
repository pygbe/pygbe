"""
Script to interpolate the wavelength dependent refraction coefficient
Uses linear interpolation

Parameters (command line arguments)
----------
file: (str) input file
wl0 : (float) initial wavelength
wlN : (float) final wavelength
N   : (int) number of wavelengths in array

Output
-----
File with interpolated values of refraction coefficient
File naming: file_inter_wl0_wlN_N
"""

import numpy
import sys

def interpolate(data, wave):

    low = numpy.where(data[:,0]>wave)[0][0]-1

    dw_d = wave-data[low,0]
    dw = data[low+1,0] - data[low,0] 
    dn = data[low+1,1] - data[low,1]
    dk = data[low+1,2] - data[low,2]

    n = data[low,1] + dw_d/dw * dn  
    k = data[low,2] + dw_d/dw * dk  

    return complex(n,k)

filename = sys.argv[1]
wl0 = float(sys.argv[2])
wlN = float(sys.argv[3])
N = int(sys.argv[4])

fileout = filename+'_inter_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]

data = numpy.loadtxt(filename)

if min(data[:,0])>wlN or max(data[:,0])<wl0:
    print 'Warning: interpolation is outside data region'

wavelength = numpy.linspace(wl0, wlN, N)
ref_index = numpy.zeros(N, dtype=complex)

for i in range(N):
    ref_index[i] = interpolate(data, wavelength[i])

save_data = numpy.zeros((N,3))
save_data[:,0] = wavelength
save_data[:,1] = ref_index.real
save_data[:,2] = ref_index.imag

numpy.savetxt(fileout, save_data)
