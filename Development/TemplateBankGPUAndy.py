import os
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

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
        self.linear0 = torch.nn.Linear(4, 175)
        self.linear1 = torch.nn.Linear(175, 97)
        self.linear2 = torch.nn.Linear(97, 46)
        self.linear_out = torch.nn.Linear(46, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear0(x))
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear_out(x)
        return x

#Upload the already trained weights and bias
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('saved_model_geo.pth', map_location=device))
model.eval()

#Template Bank Generation
start_time = time.time()

#Template Bank Generation
TemplateBank = torch.tensor([[0, 0]], device='cuda')

#for i in range(1000000):
while TemplateBank.size()[0] < 10000:
    rows, columns = TemplateBank.size()
    num_1 = torch.tensor([-1.74, -1.74], device='cuda')
    num_2 = torch.tensor([3.47, 3.47], device='cuda')
    ref_mass = torch.rand(2, device='cuda').multiply(num_2).add(num_1) #(r1 - r2) * torch.rand(a, b) + r2
    large_ref_mass = ref_mass.expand(rows, 2)
    x_data =  torch.cat((large_ref_mass, TemplateBank), 1)
    match = model(x_data) 
        
    if torch.max(match) < 0.97:
        TemplateBank = torch.cat((ref_mass.expand(1, 2), TemplateBank))
        
end_time = time.time()

print("Total time taken", end_time - start_time)
print("Size of the template bank", len(TemplateBank))


TemplateBank = to_np(TemplateBank)

rescaled_mass_1 = rescaling_the_mass(TemplateBank[:, 0])
rescaled_mass_2 = rescaling_the_mass(TemplateBank[:, 1])

mass_1_list = []
mass_2_list = []
for m1, m2 in zip(rescaled_mass_1, rescaled_mass_2): 
        mass_1, mass_2 = sorting_the_mass(m1, m2)
        mass_1_list.append(mass_1)
        mass_2_list.append(mass_2) 

plt.scatter(mass_1_list, mass_2_list, color='#5B2C6F')
#plt.scatter(to_np(TemplateBank[:, 0]), to_np(TemplateBank[:, 1]), color='#5B2C6F')
plt.xlabel('Mass 1')
plt.ylabel('Mass 2')
plt.savefig('template_bank_gpu_geo.pdf', dpi=300)