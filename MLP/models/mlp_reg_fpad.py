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

#from utils import (get_flat_dim, get_convblock_dim, get_deconvblock_padding)

from mlp_fpad import MLP_FPAD


class MLP_REG_FPAD(nn.Module):
    def __init__(self,
                 num_materials,
                 input_dims=275,
                 dense_task=[128, 128, 2],
                 dense_enc=[128, 128],
                 activation='relu',
                 bnorm=False,
                 dropout=0.0,
                 is_classifier=True):

        super(MLP_REG_FPAD, self).__init__()

        self.input_dims = input_dims
        self.dense_task = dense_task
        self.dense_enc = dense_enc
        self.dense_adv = [128, 128, num_materials]
        self.activation = activation
        self.bnorm = bnorm
        self.dropout = dropout
        self.is_classifier = is_classifier

        # Initialize encoder layers
        self.encoder = MLP_FPAD(input_dims=self.input_dims,
                           dense_dims=self.dense_enc,
                           activation=self.activation,
                           bnorm=self.bnorm,
                           dropout=self.dropout,
                           is_classifier=False)

        # Initialize task-classifier layers
        self.task_classifier = MLP_FPAD(input_dims=self.dense_enc[-1],
                                   dense_dims=self.dense_task,
                                   activation=self.activation,
                                   bnorm=self.bnorm,
                                   dropout=self.dropout,
                                   is_classifier=True)

        # Initialize adv-classifier layers
        self.adv_classifier = MLP_FPAD(input_dims=self.dense_enc[-1],
                                  dense_dims=self.dense_adv,
                                  activation=self.activation,
                                  bnorm=self.bnorm,
                                  dropout=self.dropout,
                                  is_classifier=True)

    def forward(self, x_real, x_fake):
        # forward pass encoder
        h_enc_real = self.encoder(x_real)
        h_enc_fake = self.encoder(x_fake)

        h_enc = [torch.cat([h_real, h_fake], dim=0) for h_real, h_fake in zip(h_enc_real, h_enc_fake)]

        # forward pass task-classifier
        h_task = self.task_classifier(h_enc[-1])

        # forward pass adv-classifier
        h_adv = self.adv_classifier(h_enc_fake[-1])

        return h_enc, h_task, h_adv
    
    def predict(self, x_real, x_fake):
               
        # forward pass encoder
        h_enc_real = self.encoder(x_real)
        h_enc_fake = self.encoder(x_fake)

        h_enc = [torch.cat([h_real, h_fake], dim=0) for h_real, h_fake in zip(h_enc_real, h_enc_fake)]

        # forward pass task-classifier
        h_task = self.task_classifier(h_enc[-1])
        
        probs = F.softmax(h_task[-1], dim=1)
      
        return probs


if __name__ == '__main__':

    import os
    os.getcwd()
    os.chdir("./")
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
    
    num_materials = 3
    
    model = MLP_REG_FPAD(num_materials).to(DEVICE)

    print(model)