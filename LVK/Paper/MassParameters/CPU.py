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
#from pycbc.tmpltbank.bank_output_utils import output_sngl_inspiral_table

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('CPU.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory for the input and output files 
DATA_DIR = '/users/sgreen/TemplateGeNN/LVK/Paper/MassParameters/'

#Define input location on the the training data 
TRAINING_DATASET = DATA_DIR+'100000MassTrainingDataset.csv'

#Define input location of the Standard.Scaler()
STANDARD_SCALER = DATA_DIR+'StandardScaler.bin'

#Define input location of the LearningMatch model
LEARNINGMATCH_MODEL = DATA_DIR+'LearningMatchModel.pth'

#Define ouput location of the template bank
TEMPLATE_BANK = DATA_DIR+'MassTemplateBank.hdf'

#Define the size of the template bank
SIZE = 10000

#Define functions
def sorting_the_mass(m1, m2):
    if m1>m2:
        return m1, m2
    else:
        return m2, m1

def scaling_the_mass(mass, scaler_mean, scaler_std):
    scaled_mass = (mass - scaler_mean)/ np.sqrt(scaler_std)
    return scaled_mass

def rescaling_the_mass(mass, scaler_mean, scaler_std):
    rescaled_mass = mass* np.sqrt(scaler_std) + scaler_mean
    return rescaled_mass
    
def to_np(x):
    return x.detach().numpy()

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

#mimimum_mass = torch.tensor(mimimum_mass)
#maximum_mass = torch.tensor(maximum_mass)

#Upload the already trained weights and bias
logger.info("Loading the LearningMatch model")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(LEARNINGMATCH_MODEL, map_location=device))
model.eval()

@torch.jit.script
def make_template_bank(num_temp, min_mass, max_mass):
    TemplateBank = torch.tensor([[0, 0]], device='cpu') 
    while TemplateBank.size()[0] < num_temp:
        rows, columns = TemplateBank.size()
        num_1 = torch.tensor([float(min_mass), float(min_mass)], device='cpu')
        num_2 = torch.tensor([max_mass, max_mass], device='cpu')
        ref_mass = torch.rand(2, device='cpu').multiply(num_2).add(num_1) #(r1 - r2) * torch.rand(a, b) + r2
        large_ref_mass = ref_mass.expand(rows, 2)
        x_data =  torch.cat((large_ref_mass, TemplateBank), 1)
        match = model(x_data) 
            
        if torch.max(match) < 0.97:
            TemplateBank = torch.cat((ref_mass.expand(1, 2), TemplateBank))
    return TemplateBank

#Template Bank Generation
logger.info("Generating the Template Bank using TemplateGeNN") 
start_time = time.time()
TemplateBank = make_template_bank(torch.tensor(SIZE, device='cpu'), mimimum_mass, maximum_mass)        
end_time = time.time()

TemplateBank = to_np(TemplateBank)

logger.info("Total time taken to generate a TemplateGeNN  %s", end_time - start_time)
logger.info("Size of the template bank  %s", len(TemplateBank))

rescaled_mass_1 = rescaling_the_mass(TemplateBank[:, 0])
rescaled_mass_2 = rescaling_the_mass(TemplateBank[:, 1])

with h5py.File(TEMPLATE_BANK,'w') as f_out:
    f_out['mass1'] = rescaled_mass_1
    f_out['mass2'] = rescaled_mass_2

#This is if you want an xml file
#MassBank=[]
#for m1, m2 in zip(rescaled_mass_1, rescaled_mass_2): 
#        m1, m2 = sorting_the_mass(m1, m2)
#        mass1.append(m1)
#        mass2.append(m2)
#        MassSpinBank.append([m1, m2])  
#my_bank = zip(rescaled_mass_1, rescaled_mass_2)
#output_sngl_inspiral_table('LigoMassTemplateBank.xml', my_bank, None, None)