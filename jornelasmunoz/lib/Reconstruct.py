import os, sys
import torch
import numpy as np




class CNN(torch.nn.Module):
    '''
    Define a model with only one convolutional layer and sigmoid activation function
    '''
    def __init__(self, params):
        super().__init__() 
        
        # Define model basic info
        self.params = params
        self.img_size = self.params['image_size']
        self.kernel_size = self.params['kernel_size']
        self.criterion = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr = self.params['learning_rate']) 
        self.params['model_save_path'] = f'../models/{params["kind"]}_{params["suffix"]}_recon.pth'
        
        # Define model architecture elements
        self.conv = torch.nn.Conv2d(1,1,kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        
        print("Using the following parameters:")
        for key, val in self.params.items():
            print(f"{key}: {val}")
        
    def forward(self, x):
        #output = torch.sigmoid(self.conv(x))
        # 03.20.23 Trying out a model with no activation function -- update 03.28.23 Didnt work :(
        output = self.conv(x)
        return output
    
    @staticmethod
    def load_data(params):
        # Load encoded data 
        filename_train = "../data/training_MNIST_mura"
        filename_eval  = "../data/validation_MNIST_mura"
        filename_test  = "../data/testing_MNIST_mura"

        mura_train_data = torch.load(filename_train)
        mura_eval_data = torch.load(filename_eval)
        mura_test_data = torch.load(filename_test)
        print(f"Number of elements in each dataset \nTraining: {len(mura_train_data)} \nValidation: {len(mura_eval_data)} \nTesting: {len(mura_test_data)}")

        # Create DataLoaders for each set
        loaders = {
            'train' : torch.utils.data.DataLoader(mura_train_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True, 
                                                  num_workers=0),

            'eval'  : torch.utils.data.DataLoader(mura_eval_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True, 
                                                  num_workers=0),

            'test'  : torch.utils.data.DataLoader(mura_test_data, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=False, 
                                                  num_workers=0),
        }
        
        return mura_train_data, mura_eval_data, mura_test_data, loaders


#     def _get_dataset():
        
#         # Load encoded data 
#         filename_train = "../data/training_MNIST_mura"
#         filename_eval  = "../data/validation_MNIST_mura"
#         filename_test  = "../data/testing_MNIST_mura"

#         mura_train_data = torch.load(filename_train)
#         mura_eval_data = torch.load(filename_eval)
#         mura_test_data = torch.load(filename_test)
#         print(f"Number of elements in each dataset \nTraining: {len(mura_train_data)} \nValidation: {len(mura_eval_data)} \nTesting: {len(mura_test_data)}")