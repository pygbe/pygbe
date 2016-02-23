"""
Script to run matrix version of PyGBe with different dielectric constants

Parameters (command line arguments)
----------
file_param : PyGBe parameter file
file_base:      PyGBe input file used as a base
complex_region: Region number in file_base with refraction index specified in file_refr
file_refr:      File with refractive index for each wavelength
                    At this point, all file_refr must have matching wavelengths
For more than one region with varying refraction index, alternate the region number with the 
corresponding file_refr

Output
-----
Plot of extinction cross section with wavelength
"""

import matplotlib
matplotlib.use('Agg')
import numpy
import matplotlib.pyplot as plt
import sys
import os

def createInputFile(file_base, file_new, complex_region, diel, wavelength):
# Creates input file with complex dielectric constant
# file_base     : (string) base input filename
# file_new      : (string) new input filename
# complex_region: (array of int) array with regions that have complex dielectric (based on order of base_input)
# diel          : (array of complex) array with complex dielectric constant of corresponding regions
# wavelength    : (float) wavelength of incoming field

    fn = open(file_new, 'w')
    region = -1
    for line_full in file(file_base):
        line = line_full.split()
        
        if line[0]=='FILE':
            fn.write(line_full)

        elif line[0]=='FIELD':
            region += 1
            if region in complex_region:
                index = numpy.where(complex_region==region)[0][0]
                new_line = line[0] + '\t'
                for i in range(1,len(line)):
                    if i==3:
                        new_line += str(diel[index])[1:-1] + '\t'
                    else:
                        new_line += line[i] + '\t'

                fn.write(new_line+'\n')
            else:
                fn.write(line_full)

        elif line[0]=='WAVE':
            new_line = line[0] + '\t' + line[1] + '\t' + str(wavelength)
            fn.write(new_line+'\n')
        
        else:
            fn.write(line_full)

    fn.close()


def scanOutput(filename):

    flag = 0
    Cext = []
    surf = []
    for line in file(filename):
        line = line.split()
        if len(line)>0:
            if flag == 1 and line[0] == 'Surface':
                surf.append(int(line[1][:-1]))
                Cext.append(float(line[2]))
            elif line[0] == 'Cext:':
                flag = 1
            
    return surf, Cext

file_param = sys.argv[1]
file_base = sys.argv[2]

n_file = (len(sys.argv)-2)/2
data = []
region = []
for i in range(n_file):
    i_region = 2*i+3
    i_file = 2*i+4

    region.append(int(sys.argv[i_region]))
    data.append(numpy.loadtxt(sys.argv[i_file]))

region = numpy.array(region)
diel = numpy.zeros(n_file, dtype=complex)

wavelength = data[0][:,0]*1e3 # nanometers

for i in range(len(data[0])):

    for j in range(n_file):
        ref_index = complex(data[j][i,1], data[j][i,2])
        diel[j] = ref_index*ref_index

    newFile = 'matrix_tests/input_files/sphere_complex_aux.config'
    outputFile = 'matrix_tests/output_aux'

    createInputFile(file_base, newFile, region, diel, wavelength[i])

    command = 'python matrix_tests/main_matrix.py ' + file_param + ' ' + newFile  + ' ' + ' > ' + outputFile

    os.system(command)

#   surf[i,:], Cext[i,:] = scanOutput(outputFile)
    surf_aux, Cext_aux = scanOutput(outputFile)

    if i==0:
        Cext = numpy.zeros((len(data[0]), len(surf_aux)))
        surf = numpy.zeros((len(data[0]), len(surf_aux)))

    surf[i,:] = surf_aux
    Cext[i,:] = Cext_aux


data_save = numpy.zeros((len(wavelength),len(Cext[0])+1))
data_save[:,0] = wavelength
for i in range(len(Cext[0])):
    data_save[:,i+1] = Cext[:,i]

filename = 'matrix_tests/Cext_wavelength'
numpy.savetxt(filename, data_save)

font = {'family':'serif','size':10}
fig = plt.figure(figsize=(3,2))
ax = fig.add_subplot(111)
ax.plot(wavelength, Cext[:,0])
plt.rc('font',**font)
fig.savefig('matrix_tests/Cext_wavelength.pdf', dpi=80, format='pdf')
