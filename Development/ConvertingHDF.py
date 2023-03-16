import h5py

csv_input = [line.strip().split(',') for line in open('MassSpinTemplateBank.csv').readlines()[1:]]
mass1=[]
mass2=[]
spin1z=[]
spin2z=[]
for line in csv_input:
    mass1.append(float(line[0]))
    mass2.append(float(line[1]))
    spin1z.append(float(line[2]))
    spin2z.append(float(line[3]))

with h5py.File('NewMassSpinTemplateBank8.hdf','w') as f_out:
    f_out['mass1'] = mass1
    f_out['mass2'] = mass2
    f_out['spin1z'] = spin1z
    f_out['spin2z'] = spin2z