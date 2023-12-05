import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from CNOModule import CNO
from BaselinesModules import UNet, UNetOrg
from torch.utils.data import Dataset

import scipy

from training.FourierFeatures import FourierFeatures

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

data_folder = '/scratch/PDEDatasets/CNO/new_data/'

#------------------------------------------------------------------------------

# Some functions needed for loading the Navier-Stokes data

def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)

def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real

def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
    u_down = samples_ifft(u_hat_down)
    return u_down

#------------------------------------------------------------------------------

#Load default parameters:
    
def default_param(network_properties):
    
    if "out_size" not in network_properties:
        network_properties["out_size"] = 1
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    if "retrain" not in network_properties:
         network_properties["retrain"] = 4
    
    
    return network_properties


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Poisson data:
#   From 0 to 1024 : training samples (1024)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class SinFrequencyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        
        #The file:
        if in_dist:
            self.file_data = data_folder + "PoissonData_64x64_IN.h5"
        else:
            self.file_data = data_folder + "PoissonData_64x64_OUT.h5"

        #Load normalization constants from the TRAINING set:
        self.reader = h5py.File(self.file_data, 'r')
        self.normalization_reader = h5py.File(data_folder + "PoissonData_64x64_IN.h5", 'r')
        self.min_data = self.normalization_reader['min_inp'][()]
        self.max_data = self.normalization_reader['max_inp'][()]
        self.min_model = self.normalization_reader['min_out'][()]
        self.max_model = self.normalization_reader['max_out'][()]

        self.s = s #Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024+128
            else:
                self.length = 256
                self.start = 0 
        
        #Load different resolutions
        if s!=64:
            self.file_data = data_folder + "PoissonData_NEW_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed.
        self.reader = h5py.File(self.file_data, 'r')
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid


class SinFrequency:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")

        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        

        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        torch.manual_seed(retrain)
        
        unet_mode = network_properties['mode']

        initial_channels = network_properties["initial_channels"]
        start_channels = network_properties["start_channels"]
        out_dim = network_properties["out_dim"]


        if unet_mode == "original":
            self.model = UNetOrg(n_channels = initial_channels, n_classes = out_dim, channels = start_channels).to(device)
        
        elif unet_mode == "modified":
            self.model = UNet(n_channels = initial_channels, n_classes = out_dim, start = start_channels).to(device)

        else:
            raise NotImplementedError


        #----------------------------------------------------------------------
        

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(SinFrequencyDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(SinFrequencyDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(SinFrequencyDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#---