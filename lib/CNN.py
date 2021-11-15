import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super.__init__()
        self.layer_list = torch.nn.ModuleList([
            # 8 channels in, 12 channels out, 3x3 kernel
            # 8x26x26 -> 12x24x24
            # (26 - 3 + 0 + 0)/1 + 1 ref formula for input dim
            torch.nn.Conv2d(1,32,5),
            # linear activation function
            torch.nn.ReLU(inplace=True),
            # max-pooling operator
            # 12x24x24 -> 32x12x12
            # 24/2
            torch.nn.MaxPool2d(2, 2),
            # 12 channels in, 32 channels out, 5x5 kernel
            # 32x12x12 -> 32x8x8
            # (12 - 5 + 0 + 0)/1 + 1 ref formula for input dim
            torch.nn.Conv2d(32,64,5),
            # linear activation function
            torch.nn.ReLU(inplace=True),
            # 32x8x8 -> 32x4x4 = 512
            # 8/2
            torch.nn.MaxPool2d(2, 2),
        ])
        
        # Classifier: FIX ME -> Modify hyperparameters accordingly!!!!
        self.classifier = torch.nn.Sequential(
            # Fully-Connected, 512 elements in, 2 elements out
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 160),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(160, 67),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(67, 2),
            torch.nn.Sigmoid(),
        )
    pass