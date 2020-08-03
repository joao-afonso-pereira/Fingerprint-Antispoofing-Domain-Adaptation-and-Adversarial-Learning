from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statistics import mean
import sys

dim = 128

# Number of channels in the training images.
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 128

class Reshape(nn.Module):
    def __init__(self, N, C, H, W):
        super(Reshape, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


class DISCRIMINATOR(nn.Module):
    def __init__(self):
        super(DISCRIMINATOR, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(1, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            Reshape(N=-1, C=11, H=11, W=256),
            nn.Linear(256, 1),
            Reshape(N=-1, C=1, H=11, W=11),

        )

    def forward(self, input):
        '''
        print("DISCRIMINATOR")
        y = input
        print(input.shape)
        for i in range (len(self.main)):
          y = self.main[i](y)
          print("Layer " + str(i))
          print(y.shape)
        '''
        return self.main(input)


class GENERATOR(nn.Module):
    def __init__(self, features = [64, 64, 64, 64, 64]):
        super(GENERATOR,self).__init__()
        
        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(1, features[0], kernel_size=3, stride=2, padding=1),
                                         nn.LeakyReLU(0.2, inplace=True)))
        for i in range(1, len(features)):
            conv_layers.append(nn.Sequential(nn.Conv2d(features[i-1], features[i], kernel_size=3, padding=1),
                                             nn.BatchNorm2d(features[i]),
                                             nn.LeakyReLU(0.2, inplace=True)))

        for i in range(len(features)):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(features[i], features[i-1], kernel_size=3, padding=1),
                                               nn.BatchNorm2d(features[i-1]),
                                               nn.ReLU(inplace=True)))
         
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(features[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                               nn.Tanh()))    
        
        
        self.encoder = nn.Sequential(*conv_layers)
        self.decoder = nn.Sequential(*deconv_layers)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
  
if __name__ == '__main__':

    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU...")
    else:
        DEVICE = torch.device("cpu")
        print("Running on the CPU...")
    print()

    
    D = DISCRIMINATOR().to(DEVICE)
    G = GENERATOR().to(DEVICE)

    print(D)
    print(G)

    from torchsummary import summary

    summary(D, (1, 128, 128))