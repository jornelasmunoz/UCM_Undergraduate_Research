import sys
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

desktop_path = '/Users/jocelynornelasmunoz/Desktop/Research/coded-aperture/jornelasmunoz/'
laptop_path = '/Users/jocelynornelas/iCloud Drive (Archive)/Desktop/UC Merced/Research/coded-aperture/jornelasmunoz/'
if desktop_path in sys.path[0]: sys.path.insert(0, desktop_path + 'lib/'); path = desktop_path
elif laptop_path in sys.path[0]: sys.path.insert(0, laptop_path + 'lib/'); path = laptop_path

import MURA as mura


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.kernel = 3
        # 1 input image channel, 8 output channels, 2x2 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, self.kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, self.kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = (F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 4* 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def _get_dataset(self, A,encoded=True):
        size = A.shape[0] # images will be the size of encoder
        transform_list = [
                        transforms.ToTensor(),
                        transforms.Resize(size),
                        transforms.Normalize(0, 1),
                        ]
        # select whether to get encoded data or original data
        if encoded:
            transform_list.append(transforms.Lambda(lambda x: torch.unsqueeze(torch.tensor(
                 mura.FFT_convolve(np.squeeze(x.numpy()), A,size), dtype= torch.float), 0)))
            
        train_data = datasets.MNIST(
        root = '../data/',
        train = True,                         
        transform = transforms.Compose(transform_list),
                    #     [transforms.ToTensor(),
                    #     transforms.Resize(size),
                    #     #transforms.Normalize(mean=0., std=(1/255.)),
                    #     # Apply MURA encoder
                    #     transforms.Lambda(lambda x: torch.unsqueeze(torch.tensor(
                    #         mura.FFT_convolve(np.squeeze(x.numpy()), A,size), dtype= torch.float), 0)),
                    #     transforms.Normalize(0, 1)
                    # ]), 
        download = False,            
        )
        
        test_data = datasets.MNIST(
            root = '../data/', 
            train = False, 
            transform = transforms.Compose(transform_list),
            #                 [transforms.ToTensor(),
            #                 transforms.Resize(size),
            #                 #transforms.Normalize(mean=0., std=(1/255.)),
            #                 # Apply MURA encoder
            #                 transforms.Lambda(lambda x: torch.unsqueeze(torch.tensor(
            #                     mura.FFT_convolve(np.squeeze(x.numpy()), A,size), dtype= torch.float), 0)),
            #                 transforms.Normalize(0, 1)
            #             ]) 
        )
        
        return train_data, test_data
    
    

# # Creating a DeepAutoencoder class
# class DeepAutoencoder(torch.nn.Module):
#     def __init__(self, img_size):
#         super().__init__()  
#         self.img_size = img_size
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(self.img_size * self.img_size, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 10)
#         )
          
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(10, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, self.img_size * self.img_size),
#             torch.nn.Sigmoid()
#         )
  
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded