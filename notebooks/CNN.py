import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import itertools


# data_dir = '../data/4x4'

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class CNN(nn.Module):
    def __init__(self, img_shape = 28, noise = False, conv_layers = 2, num_classes = 10):
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
        self.transform = [transforms.ToTensor(),
                          transforms.Resize(img_shape),
                          transforms.Normalize(0, 1)
                         ]
        if noise:
            self.transform.append(AddGaussianNoise(1, 1))
        
        
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
                transform = transforms.Compose(self.transform), 
                download = False,            
            )
            test_data = datasets.MNIST(
                root = root_path, 
                train = False, 
                transform = transforms.Compose(self.transform)
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
    print("Training...")
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
    
    
# FIX ME: Still modifying the test function so that it evaluates the model and returns the confusion matrix data (not the heat map)!!

def test(cnn, loaders):
    
    print("Testing...")
    cnn.eval()
#     test_acc_data = []
    pred_data = torch.tensor([])
    truth_data = torch.tensor([])
        
    # Test the model
    correct = 0
    total = 0
    for images, labels in loaders['test']:
        test_output, last_layer = cnn(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        
        pred_data = torch.cat((pred_data, pred_y))
        truth_data = torch.cat((truth_data, labels))
        
#         accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
#         test_acc_data.append(accuracy)

    stacked = torch.stack(
        (
            truth_data
            ,pred_data
        )
        ,dim=1
    ) 
    
    cmt = torch.zeros(10,10, dtype=torch.int64)
    
    for p in stacked:
        tl, pl = p.tolist()
        tl = int(tl)
        pl = int(pl)
        cmt[tl, pl] = cmt[tl, pl] + 1
        
    
    return cmt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.figure(figsize=(10,10))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')