import matplotlib
matplotlib.use('Agg')
import numpy
import matplotlib.pyplot as plt
import sys

def createInputFile(file_base, file_new, complex_region, diel):
# Creates input file with complex dielectric constant
# file_base     : base input filename
# file_new      : new input filename
# complex_region: array with regions that have complex dielectric (based on order of base_input)
# diel          : array with complex dielectric constant of corresponding regions

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
                new_line = line[:3] + '\t' + str(diel[index]) + '\t' + line[4:]
                fn.write(new_line)
        else:
            fn.write(line_full)
            
    fn.close()

file_base = sys.argv[1]

n_file = len(sys.argv)
data = []
for i in range(2, n_file):
    data.append(numpy.loadtxt(sys.argv[i]))

for i in range(len(data[0])):
    ref_index = complex(data




