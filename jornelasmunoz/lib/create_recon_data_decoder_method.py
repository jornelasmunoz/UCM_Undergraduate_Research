import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_theme()

import torchvision.transforms as transforms
from torchvision import datasets
import torch

desktop_path = '/Users/jocelynornelasmunoz/Desktop/Research/coded-aperture/jornelasmunoz/'
laptop_path = '/Users/jocelynornelas/iCloud Drive (Archive)/Desktop/UC Merced/Research/coded-aperture/jornelasmunoz/'
if desktop_path in sys.path[0]: sys.path.insert(0, desktop_path + 'lib/'); path = desktop_path
elif laptop_path in sys.path[0]: sys.path.insert(0, laptop_path + 'lib/'); path = laptop_path
print('Using path = ', path)

import MURA as mura


desired_snr = 30
# Compute MURA encoder and decoder
p = 23 # size of array (has to be prime)
A = mura.create_binary_aperture_arr(p)
G = mura.create_decoding_arr(A)

# Load MNIST data from PyTorch and
    # (1) Convert to tensor
    # (2) Resize from 28x28 to pxp 
    # (3) Normalize entries between [0,1]
train_data = datasets.MNIST(
    root = '../data/',
    train = True,                         
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(p),
                    transforms.Normalize(0, 1)
                ]), 
    download = False,            
)

test_data = datasets.MNIST(
    root = '../data/', 
    train = False, 
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(p),
                    transforms.Normalize(0, 1)
                ]) 
)

# Transform the the whole dataset into a list by concatenating training and testing
all_data_list   = list(train_data) + list(test_data)
print(f'Length of whole dataset is {len(all_data_list)}.')


# ------------------ Create new set of training, validation, and testing 
# Define list to save encoded image and original image data
mura_all_data = []
for idx in range(len(all_data_list)):
    # Get the whole dataset
    img_label_list = list(all_data_list[idx])
    
    # Get the noiseless image,  generate coded image, and add noise to coded image
    noiseless_image = all_data_list[idx][0]
    encoded_image = torch.unsqueeze(torch.tensor(
                    mura.normalize(mura.FFT_convolve(np.squeeze(noiseless_image.numpy()), A,p)), dtype= torch.float), 0)
    noisy_encoded_image = mura.add_Gaussian_noise(encoded_image, desired_snr)
    reconstructed_noisy_encoded = torch.unsqueeze(torch.tensor(
        mura.normalize(mura.FFT_convolve(np.squeeze(noisy_encoded_image.numpy()), G, p)), dtype= torch.float),0)
    
    # Data will be saved as a tuple of (reconstructed noisy encoded image, noiseless image, number, noise level) 
    # For the encoded image, do FFT convolve and then normalize images
    img_label_list[0] = reconstructed_noisy_encoded
    img_label_list[1] = noiseless_image
    img_label_list.append(all_data_list[idx][1])
    img_label_list.append(f'{desired_snr}dB')
    mura_all_data.append(tuple(img_label_list))
    
print(f'Done! Length of encoded data list is : {len(mura_all_data)}')


# Separate into training, validation and testing
# Splits of 80, 10, and 10 percent of 70000
mura_train_data = mura_all_data[:int(len(mura_all_data)*0.8)]
mura_eval_data  = mura_all_data[int(len(mura_all_data)*0.8):int(len(mura_all_data)*0.9)]
mura_test_data  = mura_all_data[int(len(mura_all_data)*0.9):] 

print(f"Number of elements in each dataset \nTraining: {len(mura_train_data)} \nValidation: {len(mura_eval_data)} \nTesting: {len(mura_test_data)}")


##------------------------- SAVE
filename_train = f"../data/training_MNIST_mura_{desired_snr}dB_reconstructed_decoder_method"
filename_eval = f"../data/validation_MNIST_mura_{desired_snr}dB_reconstructed_decoder_method"
filename_test = f"../data/testing_MNIST_mura_{desired_snr}dB_reconstructed_decoder_method"

# Save encoded data 
torch.save(mura_train_data, filename_train)
torch.save(mura_eval_data, filename_eval)
torch.save(mura_test_data, filename_test)