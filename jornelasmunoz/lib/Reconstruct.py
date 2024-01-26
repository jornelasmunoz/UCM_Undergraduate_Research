import os, sys
import torch
import numpy as np

class RECON_CNN(torch.nn.Module):
    '''
    Define a model with only one convolutional layer, NO activation function, and NO bias
    '''
    def __init__(self, params):
        super().__init__() 
        
        # Define model basic info -- gets model/data parameters from dictionary params
        self.params = params
        self.img_size = self.params['image_size']
        self.kernel_size = self.params['image_size'] if self.params['image_size'] is not None else self.params['kernel_size']
        self.params["kernel_size"] = self.kernel_size
        self.criterion = torch.nn.MSELoss() if self.params.get('loss') is None else torch.nn.L1Loss() #
        self.RUN_DIR = f'../runs/{params["model"]}/'
        self.params['model_save_path'] = self.RUN_DIR + f'{params["model"]}.pth'
    
        
        # Define model architecture elements
        # Padding is circular -- mathematical motivation
        self.conv  = torch.nn.Conv2d(1,1,kernel_size=self.kernel_size, padding='same', padding_mode='circular', bias=False)#(self.kernel_size-1)//2)
        print("Using the following parameters:")
        for key, val in self.params.items():
            print(f"{key}: {val}")
        
        # Create dir for model or load weights if they already exist
        if not os.path.exists(self.RUN_DIR):
            os.mkdir(self.RUN_DIR)
        else:
            self.load_state_dict(torch.load(self.RUN_DIR+f"{self.params['model']}.pth"))
            print("Weights loaded from {}".format(self.RUN_DIR+f"{self.params['model']}.pth"))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience=self.params['scheduler_patience'], verbose=True)
        self.total_params = sum(p.numel() for p in self.parameters())
        
    def forward(self, x):
        output = self.conv(x)
        return output

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.ones_(m.weight.data) #normal_(m.weight.data, 0.0, 0.02)
            # torch.nn.init.uniform_(m.weight.data, a=-1, b=1) #uniform


