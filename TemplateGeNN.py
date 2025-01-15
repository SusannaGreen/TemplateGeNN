#!/usr/bin/env python

# Copyright (C) 2025 Susanna M. Green and Andrew P. Lundgren 
from Model import NeuralNetwork

import numpy as np
import pandas as pd
import time
import logging

from joblib import load

import torch
import torch.nn as nn

import h5py
from pycbc.pnutils import get_imr_duration
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('TemplateGeNNFast.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory for the input and output files 
DATA_DIR = '/users/sgreen/TemplateGeNN/LVK/Paper/'

#Define input location of the LearningMatch model
LEARNINGMATCH_MODEL = DATA_DIR+'LearningMatchModel.pth'

#Define ouput location of the template bank
TEMPLATE_BANK_HDF = DATA_DIR+'TemplateGeNNFast.hdf'
TEMPLATE_BANK_CSV = DATA_DIR+'TemplateGeNNFast.csv'

#Define the size of the template bank
SIZE = 200000
MIN_MCHIRP = 5.0
MAX_MCHIRP = 20.0
REF_LAMBDA = 5.0
MIN_LAMBDA = (MAX_MCHIRP/REF_LAMBDA)**(-5/3)
MAX_LAMBDA = (MIN_MCHIRP/REF_LAMBDA)**(-5/3)
MIN_ETA = 0.1
MAX_ETA = 0.249999

#Define functions
#Define the functions
def lambda0_to_mchirp(lambda0):
    return (lambda0**(-3/5))*REF_LAMBDA

def mchirp_to_lambda0(mchirp):
    return (mchirp/REF_LAMBDA)**(-5/3)

def sorting_the_mass(m1, m2):
    if m1>m2:
        return m1, m2
    else:
        return m2, m1

def parameter_transformation(lambda0, eta):
    mchirp = lambda0_to_mchirp(lambda0)
    m1 = mass1_from_mchirp_eta(mchirp, eta)
    m2 = mass2_from_mchirp_eta(mchirp, eta)
    return m1, m2

def to_np(x):
    return x.cpu().detach().numpy()

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")

#Upload the already trained weights and bias
logger.info("Loading the LearningMatch model")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()
compiled_model = torch.compile(model)

@torch.jit.script
def make_template_bank(num_temp: int, min_lambda: float, max_lambda: float, min_eta: float, max_eta: float): 
    TemplateBank = torch.tensor([[0, 0, 0, 0]], device='cuda') 
    acceptance_tensor = torch.tensor([[-1]], device='cuda') 
    acceptance = 0
    count = 0 
    while TemplateBank.size()[0] < num_temp:
        rows, columns = TemplateBank.size()
        num_1 = torch.tensor([min_lambda, min_eta], device='cuda') #r1
        num_2 = torch.tensor([max_lambda, max_eta], device='cuda') #r2
        ref_mass = torch.rand(2, device='cuda').multiply(torch.sub(num_2, num_1)).add(num_1) #(r1 - r2) * torch.rand(a, b) + r2
        num_3 = torch.tensor([2, 2], device='cuda') #(r1 - r2)
        num_4 = torch.tensor([-1, -1], device='cuda') #r2
        ref_spin = torch.rand(1, device='cuda').multiply(num_3).add(num_4) #(r1 - r2) * torch.rand(a, b) + r2
        ref_parameters = torch.cat((ref_mass, ref_spin), 0)
        ref_parameters = torch.reshape(ref_parameters, (1,-1))
        large_ref_parameters = ref_parameters.expand(rows, 4)
        x_data =  torch.cat((large_ref_parameters, TemplateBank), 1)
        match = model(x_data) 
            
        if torch.max(match) < 0.97:
            TemplateBank = torch.cat((TemplateBank, ref_parameters), 0)
            acceptance += 1

        count += 1

        if count%1000==0:
            if acceptance == 0:
                break

            acceptance=0

    return TemplateBank

#Template Bank Generation
logger.info("Generating the Template Bank using TemplateGeNN") 
start_time = time.time()
TemplateBank = make_template_bank(SIZE, MIN_LAMBDA, MAX_LAMBDA, MIN_ETA, MAX_ETA)        
end_time = time.time()

TemplateBank = to_np(TemplateBank)

logger.info("Total time taken to generate a TemplateGeNN Template Bank is %s", end_time - start_time)
logger.info("Size of the template bank  %s", len(TemplateBank))

logger.info("Rescaling the mass")
mass_1, mass_2 = parameter_transformation(TemplateBank[:, 0], TemplateBank[:, 1])
spin_1 = TemplateBank[:, 2]
spin_2 = TemplateBank[:, 3]

#Convert the template bank into the desired file format
logger.info("Converting the template bank to a hdf5 file")

mass1=[]
mass2=[]
spin1=[]
spin2=[]
MassSpinBank=[]

mass_1, mass_2 = parameter_transformation(TemplateBank[:, 0], TemplateBank[:, 1])
spin_1 = TemplateBank[:, 2]
spin_2 = TemplateBank[:, 3]

mass1=[]
mass2=[]
spin1=[]
spin2=[]
MassSpinBank=[]
iter = 0
for m1, m2, s1, s2 in zip(mass_1, mass_2, spin_1, spin_2):
        if iter==0:
            iter += 1 
            pass
        else:
            iter += 1
            m1, m2 = sorting_the_mass(m1, m2)
            mass1.append(float(m1))
            mass2.append(float(m2))
            spin1.append(float(s1))
            spin2.append(float(s2))
            MassSpinBank.append([float(m1), float(m2), float(s1), float(s2)])
        
logger.info("Converting the template bank to a csv file")
TemplateBank =  pd.DataFrame(data=(MassSpinBank), columns=['mass1', 'mass2', 'spin1', 'spin2'])
TemplateBank.to_csv(TEMPLATE_BANK_CSV, index = False)

with h5py.File(TEMPLATE_BANK_HDF,'w') as f_out:
    f_out['approximant'] = ['IMRPhenomXAS']*len(mass1)
    f_out['f_lower'] = np.ones_like(mass1)*12
    f_out['mass1'] = mass1
    f_out['mass2'] = mass2
    f_out['spin1z'] = spin1
    f_out['spin2z'] = spin2
    f_out['template_duration'] = get_imr_duration(np.array([mass1]), np.array([mass2]), np.array([spin1]), np.array([spin2]), np.ones_like(mass1)*12, approximant='IMRPhenomD')