#!/usr/bin/env python

import numpy
import sys
import os
import glob

def scanOutput(filename):
    
    flag = 0 
    files = []
    for line in file(filename):
        line = line.split()
        if len(line)>0:
            if line[0]=='Converged':
                iterations = int(line[2])
            if line[0]=='Total' and line[1]=='elements':
                N = int(line[-1])
            if line[0]=='Totals:':
                flag = 1 
            if line[0]=='Esolv' and flag==1:
                Esolv = float(line[2])
            if line[0]=='Esurf' and flag==1:
                Esurf = float(line[2])
            if line[0]=='Ecoul' and flag==1:
                Ecoul = float(line[2])
            if line[0]=='Time' and flag==1:
                Time = float(line[2])
            if line[0]=='Reading':
                files.append(line[-1])  

    return N, iterations, Esolv, Esurf, Ecoul, Time, files

param_file = sys.argv[1]
config_file = sys.argv[2]
tilt_begin = sys.argv[3]
tilt_end = sys.argv[4]
tilt_N = sys.argv[5]
output_file = sys.argv[6]
cuda_device = sys.argv[7]

fout = open(output_file, 'w')

fout.write('\nParameter file:\n')
f = open(param_file, 'r')
fout.write(f.read())
fout.write('\n')
f.close()

name = '_'+tilt_begin+'-'+tilt_end
config_file_moved = config_file[:-7] + name + config_file[-7:]

## Create moved input file
# Works for only 1 pqr file so far (not multiple molecules) and only 1 molecular surface.
fm = open(config_file_moved, 'w')
for line_full in file(config_file):
    line = line_full.split()
    if line[0]=='FILE':
        if line[2]=='dielectric_interface':
            prot_file = line[1]
            prot_file_moved = prot_file + name
            new_line = line[0] + '\t' + prot_file_moved + '\t' + line[2] + '\n'
            fm.write(new_line)
        if line[2]=='neumann_surface' or line[2]=='dirichlet_surface':
            surf_file = line[1]
            phi_file = line[3]
            fm.write(line_full)
    elif line[0]=='FIELD' and int(line[5])>0:
        pqr_file_aux = line[7]
        pqr_file = pqr_file_aux[:-4]
        pqr_file_moved = pqr_file + name + '.pqr'
        new_line = line[0] + '\t'
        for i in range(1,len(line)):
            if i==7:
                new_line += pqr_file_moved + '\t'
            else:
                new_line += line[i] + '\t'
        fm.write(new_line+'\n')

    else:
        fm.write(line_full)

fm.close()

fout.write('Protein file: ' + prot_file + '\n')
fout.write('Sensor  file: ' + surf_file + '\n')
fout.write('Phi     file: ' + phi_file + '\n\n')

fout.close()

N = []
iterations = []
Esolv = []
Esurf = []
Ecoul = []
Time = []

til_min = float(tilt_begin)
til_max = float(tilt_end)
til_N = int(tilt_N)  

rot_min = 0.
rot_max = 360. # Non-inclusive end point
rot_N = 1   

til_angles_aux = numpy.linspace(til_min, til_max, num=til_N)  # Tilt angles (inclusive end point)
rot_angles_aux = numpy.linspace(rot_min, rot_max, num=rot_N, endpoint=False)  # Rotation angles


til_angles = []
rot_angles = []
for i in range(len(til_angles_aux)):
    if abs(til_angles_aux[i])<1e-10 or abs(til_angles_aux[i]-180)<1e-10:
        til_angles.append(til_angles_aux[i])
        rot_angles.append(rot_min)
    else:
        for j in range(len(rot_angles_aux)):
            til_angles.append(til_angles_aux[i])
            rot_angles.append(rot_angles_aux[j])


for i in range(len(til_angles)):

    cmd_move = './scripts/move_protein.py ' + prot_file + ' ' + pqr_file + ' ' + str(rot_angles[i]) + ' ' + str(til_angles[i]) + ' ' + name
    os.system(cmd_move)

    cmd_run = 'CUDA_DEVICE=' + cuda_device + ' ./main.py ' + param_file + ' ' + config_file_moved +' > output_aux_' + output_file + name
#    cmd_run = './main.py ' + param_file + ' ' + config_file_moved +' > output_aux_' + output_file  + name
    os.system(cmd_run)
    
    N_run, iterations_run, Esolv_run, Esurf_run, Ecoul_run, Time_run, files = scanOutput('output_aux_' + output_file + name)

    fout = open(output_file,'a')
    fout.write('Angles: %2.2f tilt, %2.2f rotation; \tEtot: %f kcal/mol\n'%(til_angles[i], rot_angles[i], (Esolv_run+Esurf_run)))
    fout.close()

    N.append(N_run)
    iterations.append(iterations_run)
    Esolv.append(Esolv_run)
    Esurf.append(Esurf_run)
    Ecoul.append(Ecoul_run)
    Time.append(Time_run)

    for core_file in glob.glob('*'):
        if core_file[0:5]=='core.':
            os.system('rm core.*')

Etotal = numpy.array(Esolv) + numpy.array(Esurf) + numpy.array(Ecoul)
EsurfEsolv = numpy.array(Esolv) + numpy.array(Esurf)

os.system('rm output_aux_' + output_file + name + ' ' + config_file_moved + ' ' + prot_file_moved+'.vert ' + prot_file_moved+'.face' + ' ' + pqr_file_moved)

fout = open(output_file, 'a')

fout.write('\nNumber of elements  : %i \n'%(N[0]))
fout.write('Number of iterations: max: %i, min: %i, avg: %i \n'%(max(iterations), min(iterations), int(numpy.average(iterations))))
fout.write('Coulombic energy    : %f kcal/mol \n'%(Ecoul[0]))
fout.write('Total time          : max: %fs, min: %fs, avg: %fs \n' %(max(Time), min(Time), numpy.average(Time)))

fout.write('\n                    ||              kcal/mol\n')
fout.write('   Tilt   |  Rotat  ||    Esolv      |    Esurf     |      Esurf+Esolv \n')
fout.write('------------------------------------------------------------------------------ \n')
for i in range(len(til_angles)):
    fout.write('  %3.2f  |  %3.2f || %s  | %s    | %s \n'%(til_angles[i], rot_angles[i], Esolv[i], Esurf[i], EsurfEsolv[i]))
fout.close()
