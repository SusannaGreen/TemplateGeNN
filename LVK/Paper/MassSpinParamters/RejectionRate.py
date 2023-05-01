#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew Lundgren, and Xan Morice-Atkinson 

from Model import NeuralNetwork

import numpy as np
import pandas as pd
import time
import logging

from sklearn import preprocessing
from joblib import load

import torch
import torch.nn as nn

import h5py
from pycbc.pnutils import get_imr_duration
#from pycbc.tmpltbank.bank_output_utils import output_sngl_inspiral_table

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('ReactionRate.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory for the input and output files 
DATA_DIR = '/users/sgreen/TemplateGeNN/LVK/Paper/MassSpinParamters/'

#Define input location on the the training data 
TRAINING_DATASET = DATA_DIR+'1000000MassSpinTrainingDataset.csv'

#Define input location of the Standard.Scaler()
STANDARD_SCALER = DATA_DIR+'StandardScaler.bin'

#Define input location of the LearningMatch model
LEARNINGMATCH_MODEL = DATA_DIR+'LearningMatchModel.pth'

#Define ouput location of the template bank
TEMPLATE_BANK = DATA_DIR+'3000000MassSpinTemplateBank.hdf'

#Define the size of the template bank
SIZE = 300000

#Define functions
def sorting_the_mass(m1, m2):
    if m1>m2:
        return m1, m2
    else:
        return m2, m1

def scaling_the_mass(mass, scaler_mean, scaler_std):
    return (mass - mean)/ np.sqrt(std)

def rescaling_the_mass(mass, mean, std):
    return mass* np.sqrt(std) + mean  

def to_np(x):
    return x.cpu().detach().numpy()
    
#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")

#Loading the scaler to determine the mean and variance
logger.info("Rescaling the dataset")
scaler = load(STANDARD_SCALER)
scaler_mean = float(np.mean(scaler.mean_))
scaler_std = float(np.mean(scaler.var_))
logger.info(f'IMPORTANT: The average mean of the standard scaler is {scaler_mean}')
logger.info(f'IMPORTANT: The average standard deviation of the standard scaler is {scaler_std}')

#Loading the training dataset to determine the minimum and maximum range
logger.info("Reading in the training data")
TrainingBank = pd.read_csv(TRAINING_DATASET)
logger.info("Scaling the training data")
TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']] = scaler.transform(TrainingBank[['ref_mass1', 'ref_mass2', 'mass1', 'mass2']])
mimimum_mass = float(np.mean(TrainingBank.min().values))
maximum_mass = float(np.mean(TrainingBank.max().values))
logger.info(f'IMPORTANT: The minimum scaled mass is {mimimum_mass}')
logger.info(f'IMPORTANT: The maximum scaled mass is {maximum_mass}')

#Upload the already trained weights and bias
logger.info("Loading the LearningMatch model")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()
compiled_model = torch.compile(model)

#for i in range(1000000):
@torch.jit.script
def make_template_bank(num_temp: int, min_mass: float, max_mass: float): 
    TemplateBank = torch.tensor([[0, 0, 0, 0]], device='cuda') 
    acceptance_tensor = torch.tensor([[-1]], device='cuda') 
    acceptance = 0
    count = 0 
    while TemplateBank.size()[0] < num_temp:
        rows, columns = TemplateBank.size()
        num_1 = torch.tensor([min_mass, min_mass], device='cuda') #r1
        num_2 = torch.tensor([max_mass, max_mass], device='cuda') #r2
        ref_mass = torch.rand(2, device='cuda').multiply(torch.sub(num_2, num_1)).add(num_1) #(r1 - r2) * torch.rand(a, b) + r2
        num_3 = torch.tensor([2, 2], device='cuda') #(r1 - r2)
        num_4 = torch.tensor([-1, -1], device='cuda') #r2
        ref_spin = torch.rand(2, device='cuda').multiply(num_3).add(num_4) #(r1 - r2) * torch.rand(a, b) + r2
        ref_parameters = torch.cat((ref_mass, ref_spin), 0)
        ref_parameters = torch.reshape(ref_parameters, (1,-1))
        large_ref_parameters = ref_parameters.expand(rows, 4)
        x_data =  torch.cat((large_ref_parameters, TemplateBank), 1)
        match = compiled_model(x_data) 
            
        if torch.max(match) < 0.97:
            TemplateBank = torch.cat((ref_parameters, TemplateBank), 0)
            acceptance += 1

        count += 1

        if count%100==0:
            acceptance_tensor_100 = torch.tensor([[acceptance]], device='cuda') 
            acceptance_tensor = torch.cat((acceptance_tensor, acceptance_tensor_100),0)

            if acceptance < 5:
                break

            acceptance=0

    return TemplateBank, acceptance_tensor

#Template Bank Generation
logger.info("Generating the Template Bank using TemplateGeNN") 
start_time = time.time()
TemplateBank, RejectionRate = make_template_bank(SIZE, mimimum_mass, maximum_mass)        
end_time = time.time()

TemplateBank = to_np(TemplateBank)

logger.info("Total time taken to generate a TemplateGeNN Template Bank is %s", end_time - start_time)
logger.info("Size of the template bank  %s", len(TemplateBank))

logger.info("Rescaling the mass")
rescaled_mass_1 = rescaling_the_mass(TemplateBank[:, 0],  scaler_mean, scaler_std)
rescaled_mass_2 = rescaling_the_mass(TemplateBank[:, 1],  scaler_mean, scaler_std)
spin1 = TemplateBank[:, 2]
spin2 = TemplateBank[:, 3]

#Convert the template bank into the desired file format
logger.info("Converting the template bank to a hdf5 file")

mass1=[]
mass2=[]
spin1z=[]
spin2z=[]
MassSpinBank=[]

for m1, m2, s1, s2 in zip(rescaled_mass_1, rescaled_mass_2, spin1, spin2): 
        m1, m2 = sorting_the_mass(m1, m2)
        mass1.append(m1)
        mass2.append(m2)
        spin1z.append(s1)
        spin2z.append(s2)
        MassSpinBank.append([m1, m2, s1, s2]) #Needed for the csv file format 

logger.info("Converting the template bank to a csv file")
TemplateBank =  pd.DataFrame(data=(MassSpinBank), columns=['mass1', 'mass2', 'spin1', 'spin2'])
TemplateBank.to_csv('3000000MassSpinTemplateBank.csv', index = False)

with h5py.File(TEMPLATE_BANK,'w') as f_out:
    f_out['approximant'] = ['IMRPhenomXAS']*len(mass1)
    f_out['f_lower'] = np.ones_like(mass1)*12
    f_out['mass1'] = mass1
    f_out['mass2'] = mass2
    f_out['spin1z'] = spin1z
    f_out['spin2z'] = spin2z
    f_out['template_duration'] = get_imr_duration(np.array([mass1]), np.array([mass2]), np.array([spin1z]), np.array([spin2z]), np.ones_like(mass1)*12, approximant='IMRPhenomD')

acceptance_int = to_np(acceptance_tensor)
logger.info("Size of the template bank  %s", acceptance_int)
#logger.info("Converting the template bank to a xml file")
#my_bank = zip(rescaled_mass_1, rescaled_mass_2, spin1, spin2)
#output_sngl_inspiral_table('MassSpinTemplateBank.xml', my_bank, None, None)

