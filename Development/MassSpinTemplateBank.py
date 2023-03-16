import os
import numpy as np
import pandas as pd
import time
import h5py

import torch
import torch.nn as nn

#Check that Pytorch recognises there is a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def sorting_the_mass(m1, m2):
    if m1>m2:
        return m1, m2
    else:
        return m2, m1

def scaling_the_mass(mass):
    scaled_mass = (mass - 51)/ np.sqrt(799)
    return scaled_mass

def rescaling_the_mass(mass):
    rescaled_mass = mass* np.sqrt(799) + 51
    return rescaled_mass
    
def to_np(x):
    return x.cpu().detach().numpy()

#Define the Neural Network
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear0 = torch.nn.Linear(8, 275)
        self.linear1 = torch.nn.Linear(275, 210)
        self.linear2 = torch.nn.Linear(210, 248)
        self.linear3 = torch.nn.Linear(248, 76)
        self.linear_out = torch.nn.Linear(76, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = self.linear_out(x)
        return x

#Upload the already trained weights and bias
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('saved_model_150.pth', map_location=device))
model.eval()

#for i in range(1000000):
@torch.jit.script
def make_template_bank(num_temp):
    TemplateBank = torch.tensor([[0, 0, 0, 0]], device='cuda') 
    while TemplateBank.size()[0] < num_temp:
        rows, columns = TemplateBank.size()
        num_1 = torch.tensor([-1.74, -1.74], device='cuda')
        num_2 = torch.tensor([3.47, 3.47], device='cuda')
        ref_mass = torch.rand(2, device='cuda').multiply(num_2).add(num_1) #(r1 - r2) * torch.rand(a, b) + r2
        num_3 = torch.tensor([2, 2], device='cuda')
        num_4 = torch.tensor([-1, -1], device='cuda')
        ref_spin = torch.rand(2, device='cuda').multiply(num_3).add(num_4)
        ref_parameters = torch.cat((ref_mass, ref_spin), 0)
        ref_parameters = torch.reshape(ref_parameters, (1,-1))
        large_ref_parameters = ref_parameters.expand(rows, 4)
        x_data =  torch.cat((large_ref_parameters, TemplateBank), 1)
        match = model(x_data) 
            
        if torch.max(match) < 0.97:
            TemplateBank = torch.cat((ref_parameters, TemplateBank), 0)
    return TemplateBank

#Template Bank Generation
start_time = time.time()
TemplateBank = make_template_bank(torch.tensor(1000))        
end_time = time.time()

print("Total time taken to generate a TemplateGeNN", end_time - start_time)
print("Size of the template bank", len(TemplateBank))

TemplateBank = to_np(TemplateBank)

rescaled_mass_1 = rescaling_the_mass(TemplateBank[:, 0])
rescaled_mass_2 = rescaling_the_mass(TemplateBank[:, 1])
spin1 = TemplateBank[:, 2]
spin2 = TemplateBank[:, 3]

#Getting the Data in the file format required to generate a hdf file
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
        MassSpinBank.append([m1, m2, s1, s2])

with h5py.File('100MassSpinTemplateBank.hdf','w') as f_out:
    f_out['mass1'] = mass1
    f_out['mass2'] = mass2
    f_out['spin1z'] = spin1z
    f_out['spin2z'] = spin2z

TemplateBank =  pd.DataFrame(data=(MassSpinBank), columns=['mass1', 'mass2', 'spin1', 'spin2'])
TemplateBank.to_csv('100MassSpinTemplateBank.csv', index = False)