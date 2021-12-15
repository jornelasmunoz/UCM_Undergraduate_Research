import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super.__init__()
        '''
        Input: 
        First convolution: 
        First MaxPool:
        Second 
        '''
        self.layer_list = torch.nn.ModuleList([
            # 8 channels in, 12 channels out, 5x5 kernel
            # 1x28x28 -> 32x24x24
            # (26 - 3 + 0 + 0)/1 + 1 ref formula for input dim
            torch.nn.Conv2d(1,32,5),
            # linear activation function
            torch.nn.ReLU(inplace=True),
            # max-pooling operator
            # 32x24x24 -> 32x12x12
            # ((24 - 2)/2) + 1 = 12
            torch.nn.MaxPool2d(2, 2),
            # 12 channels in, 32 channels out, 5x5 kernel
            # 32x12x12 -> 64x8x8
            # (12 - 5 + 2*0)/1 + 1 = 8 ref formula for input dim
            torch.nn.Conv2d(32,64,5),
            # linear activation function
            torch.nn.ReLU(inplace=True),
            # 64x8x8 -> 64x4x4 = 1024
            # (8 - 2)/2 + 1 = 4
            torch.nn.MaxPool2d(2, 2),
        ])
        
        self.classifier = torch.nn.Sequential(
            # Fully-Connected, 1024 elements in, 10 elements out
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 160),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(160, 67),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(67, 10),
            torch.nn.Softmax(),
        )
        
        
        def forward(self, x):
            pass
        