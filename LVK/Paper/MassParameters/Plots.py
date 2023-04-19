#!/usr/bin/env python

# Copyright (C) 2023 Susanna M. Green, Andrew P. Lundgren, and Xan Morice-Atkinson 

from Model import NeuralNetwork

import numpy as np
import pandas as pd
import seaborn as sns
import logging
import time 

from sklearn import preprocessing
from joblib import load

import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('Plots.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/TemplateGeNN/LVK/Paper/MassParameters/'

#Define location of the templatebank 
TEMPLATE_BANK = DATA_DIR+r'TemplateBank.hdf'

#Defining the location of the outputs
HISTOGRAM = DATA_DIR+'TemplateBankHistogram.pdf'

#Reading the template bank
logging.info("Reading in the template bank")
with h5py.File(TEMPLATE_BANK, "r") as f:
    mass_1 = f['mass1'][()]  
    mass_2 = f['mass2'][()]

#Creates a histogram of the errors
logger.info("Creating a 2D histogram") 
plt.figure(figsize=(9, 9))
plt.hist2d(mass_1, mass_2)
plt.xlabel('Mass 1 ($M_{\odot}$)')
plt.ylabel('Mass 2 ($M_{\odot}$)')
plt.legend(loc='upper left')
plt.savefig(HISTOGRAM, dpi=300, bbox_inches='tight')
