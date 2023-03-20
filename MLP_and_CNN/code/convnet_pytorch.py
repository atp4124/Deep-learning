"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

class PreActBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        )

    def forward(self, x):
        z = self.net(x)
        out = z + x
        return out

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        super(ConvNet, self).__init__()
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """

        self.net = nn.Sequential(nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, stride=1),
                                 PreActBlock(64, 64),
                                 nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                 PreActBlock(128, 128),
                                 PreActBlock(128, 128),
                                 nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                 PreActBlock(256, 256),
                                 PreActBlock(256, 256),
                                 nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                 PreActBlock(512, 512),
                                 PreActBlock(512, 512),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                 PreActBlock(512, 512),
                                 PreActBlock(512, 512),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(512, n_classes))


    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """

        out = self.net(x)
        return out
