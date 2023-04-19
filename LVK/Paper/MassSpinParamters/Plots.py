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

import plotly
import plotly_express as px

#Set-up the logging 
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO) # set log level 

file_handler = logging.FileHandler('Plots.log') # define file handler and set formatter
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) # add file handler to logger

#Define directory of the input and output files 
DATA_DIR = '/users/sgreen/TemplateGeNN/LVK/Paper/MassSpinParameters/'

#Define location of the templatebank 
TEMPLATE_BANK = DATA_DIR+r'TemplateBank.hdf'

#Defining the location of the outputs
HISTOGRAM = DATA_DIR+'TemplateBankHistogram.pdf'

#Define functions
def chi_effective(m1, m2, s1, s2): 
    chi_eff_numerator = m1*s1 + m2*s2
    m_total = m1+m2
    return chi_eff_numerator/m_total

def chirp_mass(m1, m2):
    chirp_mass_numerator = (m1*m2)**(3/5)
    chirp_mass_denominator = (m1+m2)**(1/5)
    return chirp_mass_numerator/chirp_mass_denominator

#Reading the template bank
logging.info("Reading in the template bank")
with h5py.File(TEMPLATE_BANK, "r") as f:
    mass_1 = f['mass1'][()]  
    mass_2 = f['mass2'][()]
    spin_1 = f['spin1'][()]
    spin_2 = f['spin2'][()]

#Calculating chirp-mass and chi-effective
logging.info("Calculating chirp-mass and chi-effective")
chi_eff = []
m_chirp = []
for m1, m2, s1, s2 in zip(mass_1, mass_2, spin_1, spin_2):
    chi_eff.append(chi_effective(m1, m2, s1, s2))
    m_chirp.append(chirp_mass(m1, m2))

#Creates a histogram of the errors
logger.info("Creating a 2D histogram") 
plt.figure(figsize=(9, 9))
plt.hist2d(m_chirp, chi_eff)
plt.xlabel('$\mathcal{M}$)')
plt.ylabel('$\chi_{\textit{eff}}$')
plt.legend(loc='upper left')
plt.savefig(HISTOGRAM, dpi=300, bbox_inches='tight')

