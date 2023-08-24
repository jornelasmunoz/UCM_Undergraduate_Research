import sys
import numpy as np
import random
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
import params_functions as pf

class classification_cnn(nn.Module):
    def __init__(self,params):
        super(classification_cnn, self).__init__()
        self.params = pf.model_params_defaults(params, kind = 'classification')
        self.params["conv_sizes_list"] = []
        self.params["linear_sizes_list"] = []
        self.params["pool_size"] = 2
        self.kernel = self.params.get("kernel_size") if self.params.get("kernel_size") is not None else 3
        
        if self.params["image_size"] == 23:
            self.params["conv_sizes_list"] = [23, 21, 10, 8, 4]
            self.params["linear_sizes_list"] = [512, 256, 128, 64, 10]
            
        elif self.params["image_size"] == 12:
            self.params["conv_sizes_list"] = [12, 10, 5, 3, 1]
            self.params["linear_sizes_list"] = [64, 32, 16, 8, 10]
            
        # 1 input image channel, 8 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, self.kernel)
        self.pool = nn.MaxPool2d(self.params["pool_size"],self.params["pool_size"])
        self.conv2 = nn.Conv2d(8, 16, self.kernel)
        self.pool = nn.MaxPool2d(self.params["pool_size"],self.params["pool_size"])
        self.fc1 = nn.Linear(16 * self.params["conv_sizes_list"][-1] * self.params["conv_sizes_list"][-1], self.params["linear_sizes_list"][0])
        self.fc2 = nn.Linear(self.params["linear_sizes_list"][0], self.params["linear_sizes_list"][1])
        self.fc3 = nn.Linear(self.params["linear_sizes_list"][1], self.params["linear_sizes_list"][2])
        self.fc4 = nn.Linear(self.params["linear_sizes_list"][2], self.params["linear_sizes_list"][3])
        self.fc5 = nn.Linear(self.params["linear_sizes_list"][3], self.params["linear_sizes_list"][4])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = (F.relu(self.conv3(x)))
        x = x.view(-1, 16 * self.params["conv_sizes_list"][-1] * self.params["conv_sizes_list"][-1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def _get_clean_dataset(self, params):
        dataset_name = params['dataset'].split("_")[0]
        data_mapper = datasets.MNIST if "MNIST".upper() in dataset_name else datasets.CIFAR10
        data_root = '../data/' if "MNIST".upper() in dataset_name else '../data/CIFAR10'
        size = params['image_size'] # images will be the size of encoder
        transform_list = [
                        transforms.ToTensor(),
                        transforms.Resize(size),
                        transforms.Normalize(0, 1),
                        ]
        if "CIFAR".upper() in dataset_name: transform_list.append(transforms.Grayscale())
#         # select whether to get encoded data or original data
#         if encoded:
#             transform_list.append(transforms.Lambda(lambda x: torch.unsqueeze(torch.tensor(
#                  mura.FFT_convolve(np.squeeze(x.numpy()), params['A'],size), dtype= torch.float), 0)))
            
        train_data = data_mapper(
        root = data_root,
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
        
        test_data = data_mapper(
            root = data_root, 
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
        
        # split data into train, validation, test
        data_len = len(train_data + test_data)
        all_data = train_data + test_data
        random_indices = [*range(data_len)]
        random.shuffle(random_indices)
        split_idx_1 = int(0.8 * data_len)
        split_idx_2 = int(0.9 * data_len)
        
        # split into 80 train, 10, validation, 10 test
        train_data = torch.utils.data.Subset(all_data, random_indices[:split_idx_1])
        validation_data = torch.utils.data.Subset(all_data, random_indices[split_idx_1:split_idx_2])
        test_data = torch.utils.data.Subset(all_data, random_indices[split_idx_2:])
        
        # Create DataLoaders for each set
        loaders = {
            'train' : torch.utils.data.DataLoader(train_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True, 
                                                  num_workers=0),
            'eval' : torch.utils.data.DataLoader(validation_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True, 
                                                  num_workers=0),
            'test'  : torch.utils.data.DataLoader(test_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=False, 
                                                  num_workers=0),
        }
        return train_data, test_data, validation_data, loaders
    
    @staticmethod
    def load_encoded_data(params):
        # Updated 04/24/23 to be able to load noisy data
        '''
        Loads MNIST MURA encoded data
        
        Inputs:
            params - dictionary. Must contain keys "batch_size" and "dataset"
        Outputs:
            encoded_train_data - Array of tensors containing the training dataset. Images are encoded with given noise level 
                                (or noiseless)
            encoded_eval_data  - Array of tensors for validation set 
            encoded_test_data  - Array of tensors for test set 
            loaders            - dictionary with PyTorch Dataloaders as values
        
        '''
        dataset = 'MNIST' if 'MNIST'.upper() in params['dataset'] else 'CIFAR10'
        # Load reconstructed data 
        filename_train = f"../data/{dataset}/training_{params['dataset']}"
        filename_eval = f"../data/{dataset}/validation_{params['dataset']}"
        filename_test = f"../data/{dataset}/testing_{params['dataset']}"
        print(f"Loading dataset from: {filename_train}")
        recon_train_data = torch.load(filename_train)
        recon_eval_data = torch.load(filename_eval)
        recon_test_data = torch.load(filename_test)
        #print(f"Number of elements in each dataset \nTraining: {len(recon_train_data)} \nValidation: {len(recon_eval_data)} \nTesting: {len(recon_test_data)}")

        # Create DataLoaders for each set
        loaders = {
            'train' : torch.utils.data.DataLoader(recon_train_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True, 
                                                  num_workers=0),

            'eval'  : torch.utils.data.DataLoader(recon_eval_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True, 
                                                  num_workers=0),

            'test'  : torch.utils.data.DataLoader(recon_test_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=False, 
                                                  num_workers=0),
        }
        
        return recon_train_data, recon_eval_data, recon_test_data, loaders

    
    # ----------------------------------------------------------------------------
    #
    #                           HELPER FUNCTIONS
    #
    # ----------------------------------------------------------------------------
    def evaluate_model(self,model,loaders):
        # Initialize counters
        correct = 0
        total = 0

        # Initialize lists for storage
        incorrect_examples = []
        predicted_all = []
        labels_all = []

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(loaders['test']):
                # ----------- get images and labels -----------
                if len(data) == 3:
                    # MNIST_mura dataset: [encoded image, original image, digit]
                    images, _, labels = data
                elif len(data) == 4:
                    # MNIST_mura_{SNR}dB dataset: [encoded (noisy or noiseless) image, original image, digit, noise level]
                    images, _, labels, _ = data
                elif len(data) == 2: 
                    # Clean data -- out of the box
                    images, labels = data
                else: 
                    raise Exception("Make sure you are loading the correct data")

                # ----------- calculate outputs/preds -----------
                # calculate outputs by running images through the network (done in batches)
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                idxs_mask = torch.nonzero(predicted != labels) #((predicted == labels) == False).nonzero()
                for single_sample in idxs_mask:
                    incorrect_examples.append([np.squeeze(images[single_sample].numpy()), 
                                               labels[single_sample].numpy()[0], 
                                               predicted[single_sample].numpy()[0]])
                predicted_all.append(predicted.tolist())
                labels_all.append(labels.tolist())

        print(f'Accuracy: {100 * (correct / total):.3f} %')

        predicted_all = list(np.concatenate(predicted_all).flat) 
        labels_all = list(np.concatenate(labels_all).flat) 
        
        return predicted_all, labels_all, incorrect_examples
    
    
    
    
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