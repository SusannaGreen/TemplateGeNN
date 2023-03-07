import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from pycbc.filter import matchedfilter
from pycbc.psd.analytical import aLIGOaLIGO175MpcT1800545
from pycbc.waveform import get_fd_waveform, get_td_waveform

#Functions
def sorting_the_mass(m1, m2):
    if m1>m2:
        return m1, m2
    else:
        return m2, m1

#Defining the PSD
sample_rate = 4096
tlen = 128
delta_f = 1.0 / tlen
psd = aLIGOaLIGO175MpcT1800545(1+tlen*sample_rate//2, delta_f=delta_f, low_freq_cutoff=12)

#Template Bank Generation
start_time = time.time()
TemplateBank = []

while len(TemplateBank) < 1000:
    ref_mass = np.random.uniform(2.0, 100, size=(2))
    
    accept = True
    for mass in TemplateBank:
        template_reference, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=ref_mass[0], mass2=ref_mass[1], delta_f=delta_f, f_lower=12)
        template, _ = get_fd_waveform(approximant='IMRPhenomXAS', mass1=mass[0], mass2=mass[1], delta_f=delta_f, f_lower=12)
        template_reference.resize(len(psd))
        template.resize(len(psd))
        match, Index = template.match(template_reference, psd=psd, low_frequency_cutoff=15)

        if match > 0.97:
            accept = False
            break

    if accept == True:
        mass_1, mass_2 = sorting_the_mass(ref_mass[0], ref_mass[1]) 
        TemplateBank.append([mass_1, mass_2])

end_time = time.time()

print("Total time taken to generate a PyCBC template length", end_time - start_time)
print("Size of the template bank", len(TemplateBank))

matplotlib.rcParams.update({'font.size': 14})
plt.scatter(TemplateBank[:, 0], TemplateBank[:, 1], color='#5B2C6F')
plt.xlabel('Mass 1')
plt.ylabel('Mass 2')
plt.savefig('template_bank_pycbc.pdf', dpi=300)