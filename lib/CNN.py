import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np


data_dir = '../data/4x4'

class CNN(nn.Module):
    def __init__(self, img_shape = 28, conv_layers = 2, num_classes = 10):
        super(CNN, self).__init__()
        
#         self.conv_count = 0   #keeps track of the number of times a conv layer is added
        self.img_shape = img_shape   
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.root_path = '../data/28x28'
        self.ker_size = 0
        self.padding = 0
        self.out_channels = 0
        self.linear_shape = [1, img_shape, img_shape]
        self.conv = []
        
        
        def conv_out_size(linear_shape, img_shape, conv):
            linear_shape[0] = conv[0].out_channels
            linear_shape[1] = int(( (img_shape - conv[0].kernel_size[0] + 2*conv[0].padding[0]) / conv[0].stride[0]) + 1)
            img_shape = linear_shape[1]
            linear_shape[2] = linear_shape[1]

            return linear_shape, img_shape


        def pooling_out_size(linear_shape, img_shape, conv):
            linear_shape[1] = int(( (img_shape - conv[2].kernel_size) / conv[2].stride) + 1)
            img_shape = linear_shape[1]
            linear_shape[2] = linear_shape[1]

            return linear_shape, img_shape

        def flatten(linear_shape):
            return linear_shape[0] * linear_shape[1] * linear_shape[2]
        
        
        def load_data(img_shape, root_path):
            train_data = datasets.MNIST(
                root = root_path,
                train = True,                         
                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(img_shape),
                                transforms.Normalize(0, 1)
                            ]), 
                download = False,            
            )
            test_data = datasets.MNIST(
                root = '../data/28x28', 
                train = False, 
                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(img_shape),
                                transforms.Normalize(0, 1)
                            ]) 
            )
            
            return train_data, test_data
        
        
        # Loading training and testing data
        self.train_data, self.test_data = load_data(self.img_shape, self.root_path)
        
        self.loaders = {
            'train' : torch.utils.data.DataLoader(self.train_data, 
                                                  batch_size=100, 
                                                  shuffle=True, 
                                                  num_workers=1),

            'test'  : torch.utils.data.DataLoader(self.test_data, 
                                                  batch_size=100, 
                                                  shuffle=True, 
                                                  num_workers=1),
        }
        
        if self.img_shape == 28:
            self.out_channels = 16
            self.ker_size = 5
        
        elif self.img_shape == 14:
            self.out_channels = 8
            self.ker_size = 3
            
        elif self.img_shape == 7:
            self.out_channels = 4
            self.ker_size = 2
            
        elif self.img_shape == 4:
            self.out_channels = 4
            self.ker_size = 3
            self.padding = 1
        
        for i in range(conv_layers):
#             ker_size = 5
#             padding = 0
            if not self.conv:
                layer = nn.Sequential(         
                    nn.Conv2d(
                        in_channels=1,              
                        out_channels=self.out_channels,            
                        kernel_size=self.ker_size,              
                        stride=1,                   
                        padding=self.padding,                  
                    ),                              
                    nn.ReLU(),                      
                    nn.MaxPool2d(kernel_size=2),    
                )
                
            else:
                layer = nn.Sequential(         
                    nn.Conv2d(
                        in_channels = self.conv[i-1][0].out_channels,              
                        out_channels = self.conv[i-1][0].out_channels * 2,            
                        kernel_size = self.ker_size,              
                        stride = 1,                   
                        padding=self.padding,                  
                    ),                              
                    nn.ReLU(),                      
                    nn.MaxPool2d(kernel_size=2),    
                )
            
            self.conv.append(layer)
#             self.conv_count += 1
            
            self.linear_shape, self.img_shape = conv_out_size(self.linear_shape, self.img_shape, self.conv[i])
            self.linear_shape, self.img_shape = pooling_out_size(self.linear_shape, self.img_shape, self.conv[i])
            
#             if self.img_shape <= 0 and self.ker_size >= 2:
#                 self.ker_size -= 1
#                 i = -1
#                 self.img_shape = img_shape
#                 print("executed!!")
            
#             print("ker_size = ", self.ker_size)
            
#             elif img_shape == 0 and padding == 0:
#                 padding += 1
#                 i = -1
#                 self.img_shape = img_shape
#                 print(img_shape)
            
            continue
        
        # input to Linear layer MUST be an integer!!
        self.out_shape = int(flatten(self.linear_shape))
        self.out = nn.Linear(self.out_shape, self.num_classes)
        
        
        
        
        
    def forward(self, x):
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            
        # flatten the output of conv2 to (batch_size, 32 * 4 * 4)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization
    
    

def train(cnn, loaders, num_epochs=10):
    loss_func = nn.CrossEntropyLoss()   
    optimizer = torch.optim.Adam(cnn.parameters(), lr= 1e-5)
    
    cnn.train()
    
    train_acc_data = []
    loss_data = []
    
    # Train the model
    total_step = len(loaders['train'])    
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # measure accuracy and record loss
            train_output, last_layer = cnn(images)
            pred_y = torch.max(train_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
#             if (i+1) % 100 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
#                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), accuracy))
                
            if (i+1) % 600 == 0:
                train_acc_data.append(accuracy)
                loss_data.append(loss)
            pass
        
        pass
    
    print('Done training')
    
    return train_acc_data, loss_data
    
def test(cnn, loaders):
    y_pred = []
    y_true = []

    for inputs, labels in loaders['test']:
            output = cnn(inputs) # Feed Network

            output = (torch.max(torch.exp(output[0]), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                         columns = [i for i in classes])
#     plt.figure(figsize = (12,7))
#     sn.heatmap(df_cm, annot=True)
#     plt.savefig('../models/confusion_matrix.png')
    
    print("Done testing")
    return df_cm