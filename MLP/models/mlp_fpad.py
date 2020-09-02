'''
BASED ON ANA FILIPA SEQUEIRA'S PREVIOUS WORK
'''

import math
import sys

sys.path.insert(0, '../data/')
sys.path.insert(0, '../layers/')
sys.path.insert(0, '../utils/')

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ShuffleSplit
from torch import Tensor
from torch.nn import functional as F

from mlp_layers import BasicDenseLayer

class MLP_FPAD(nn.Module):
    def __init__(self,
                 input_dims=275,
                 dense_dims=[128, 128, 2],
                 activation='relu',
                 bnorm=False,
                 dropout=0.0,
                 is_classifier=True):

        super(MLP_FPAD, self).__init__()

        self.input_dims = input_dims
        self.dense_dims = dense_dims
        self.activation = activation
        self.bnorm = bnorm
        self.dropout = dropout
        self.n_layers = len(self.dense_dims)
        self.is_classifier = is_classifier
        if self.is_classifier:
            self.n_layers = self.n_layers - 1

        # Initialize encoder layers
        self.create_dense_layers()

    def create_dense_layers(self):
        # first dense layer
        dense_list = nn.ModuleList([
            BasicDenseLayer(in_features=self.input_dims,
                            out_features=self.dense_dims[0],
                            bnorm=self.bnorm,
                            activation=self.activation,
                            dropout=self.dropout)
        ])

        # remaining dense layers
        dense_list.extend([BasicDenseLayer(in_features=self.dense_dims[l-1],
                                           out_features=self.dense_dims[l],
                                           bnorm=self.bnorm,
                                           activation=self.activation,
                                           dropout=self.dropout)
                          for l in range(1, self.n_layers)])

        # Last dense layer
        if self.is_classifier:
            dense_list.append(BasicDenseLayer(in_features=self.dense_dims[-2],
                                              out_features=self.dense_dims[-1],
                                              bnorm=self.bnorm,
                                              activation='linear'))

        self.denseBlock = nn.Sequential(*dense_list)

    def forward(self, x):
        # get the activations of each layer
        h_list = ()
        for layer in range(len(self.dense_dims)):
            x = self.denseBlock[layer](x)
            h_list += x,
        return h_list
    
    def predict(self, x):
        #probabilities of each class
        h_list = self.forward(x)
        probs = F.softmax(h_list[-1], dim=1)
      
        return probs


if __name__ == '__main__':
    
    import os
    os.getcwd()
    os.chdir("../")
    os.getcwd()
    print(os.getcwd())

    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU...")
    else:
        DEVICE = torch.device("cpu")
        print("Running on the CPU...")
    print()
    
    input_dims = 275
    model = MLP_FPAD(input_dims=input_dims).to(DEVICE)

    print(model)
