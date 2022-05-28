# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
#from sklearn.model_selection import train_test_split

# for evaluating the model
#from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
import torchvision
import torch.nn as nn 
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained = True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, 1)
        self.fc2 = nn.Linear(num_ftrs, 1)
        self.fc3 = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.resnet18(x)
        shape = self.fc1(x)
        color = self.fc2(x)
        area = self.fc3(x)
        return {'shape': shape, 'color': color, 'area': area}

'''
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(3),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.25),
            # Defining another 2D convolution layer
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(3),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.25)
        )

        self.shape_layer = Sequential(
            #Linear(3*24*24, 576),
            ReLU(),
            Dropout(0.25),
            Linear(1728, 1)
        )
        self.color_layer = Sequential(
            #Linear(3*24*24, 576),
            ReLU(),
            Dropout(0.25),
            Linear(1728, 1)
        )
        self.area_layer = Sequential(
            #Linear(3*24*24, 576),
            ReLU(),
            Dropout(0.25),
            Linear(1728, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        #print(x.shape)
        x = self.cnn_layers(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        shape = self.shape_layer(x)
        color = self.color_layer(x)
        area = self.area_layer(x)
        return {'shape': shape, 'color': color, 'area': area}






self.color_layer = Sequential(
            Linear(48*48, 24*24),
            ReLU(inplace=True),
            Dropout(0.25),
            Linear(24*24, 3)
        )
        self.area_layer = Sequential(
            Linear(96 * 96, 48 * 48),
            ReLU(inplace=True),
            BatchNorm2d(2),
            Dropout(0.25),
            Linear(48 * 48, 1)
        )    
'''