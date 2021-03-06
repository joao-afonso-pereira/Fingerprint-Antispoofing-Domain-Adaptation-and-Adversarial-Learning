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

from cnn2_layers import BasicConvLayer, BasicDenseLayer

N_CLASSES = 2  
CHANNELS = 1        

CONV_FILTERS = [64, 64, 128, 128, 256, 256, 256, 256]

N_CONV = len(CONV_FILTERS)


MAX_POOL = [False, True, False, True, False, False, False, True]

K_SIZES = [3]*N_CONV
STRIDES = [1]*N_CONV
PADDINGS = [1]*N_CONV

TASK_DIMS = [4096, 4096, 1000, 2]
ADV_DIMS = [4096, 4096, 1000, 3]

DROPOUT = .5
BATCH_NORM = True

N_CONV = len(CONV_FILTERS)
N_FC = len(TASK_DIMS)

class CNN_REG(nn.Module):
    def __init__(self,
                 activation='relu',
                 bnorm=False,
                 dropout=0.0):

        super(CNN_REG, self).__init__()

        self.activation = activation
        self.bnorm = bnorm
        self.dropout = dropout

        # Initialize conv layers
        self.create_conv_layers()
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Initialize fc layers
        self.create_task_classifier()
        self.create_adv_classifier()



    def create_conv_layers(self):
        # first conv layer
        conv_list = nn.ModuleList([
            BasicConvLayer(in_channels=CHANNELS,
                       out_channels=CONV_FILTERS[0],
                       kernel_size=K_SIZES[0],
                       stride=STRIDES[0],
                       bnorm=True,
                       activation=self.activation,
                       max_pool=MAX_POOL[0])
        ])

        # remaining conv layers
        conv_list.extend([BasicConvLayer(in_channels=CONV_FILTERS[l-1],
                       out_channels=CONV_FILTERS[l],
                       kernel_size=K_SIZES[l],
                       stride=STRIDES[l],
                       bnorm=True,
                       activation=self.activation,
                       max_pool=MAX_POOL[l])
                          for l in range(1, N_CONV)])


        self.feature_extractor = nn.Sequential(*conv_list)

    def create_task_classifier(self):
        
        # first dense layer
        dense_list = nn.ModuleList([
            BasicDenseLayer(in_features=CONV_FILTERS[-1] * 7 * 7,
                            out_features=TASK_DIMS[0],
                            bnorm=self.bnorm,
                            activation=self.activation,
                            dropout=self.dropout)
        ])

        # remaining dense layers
        dense_list.extend([BasicDenseLayer(in_features=TASK_DIMS[l-1],
                                           out_features=TASK_DIMS[l],
                                           bnorm=self.bnorm,
                                           activation=self.activation,
                                           dropout=self.dropout)
                          for l in range(1, N_FC-1)])

        # Last dense layer
        dense_list.append(BasicDenseLayer(in_features=TASK_DIMS[-2],
                                              out_features=TASK_DIMS[-1],
                                              bnorm=self.bnorm,
                                              activation='linear'))

        self.task_classifier = nn.Sequential(*dense_list)

    def create_adv_classifier(self):
        
        # first dense layer
        dense_list = nn.ModuleList([
            BasicDenseLayer(in_features=CONV_FILTERS[-1] * 7 * 7,
                            out_features=ADV_DIMS[0],
                            bnorm=self.bnorm,
                            activation=self.activation,
                            dropout=self.dropout)
        ])

        # remaining dense layers
        dense_list.extend([BasicDenseLayer(in_features=ADV_DIMS[l-1],
                                           out_features=ADV_DIMS[l],
                                           bnorm=self.bnorm,
                                           activation=self.activation,
                                           dropout=self.dropout)
                          for l in range(1, N_FC-1)])

        # Last dense layer
        dense_list.append(BasicDenseLayer(in_features=ADV_DIMS[-2],
                                              out_features=ADV_DIMS[-1],
                                              bnorm=self.bnorm,
                                              activation='linear'))

        self.adv_classifier = nn.Sequential(*dense_list)

    def forward(self, x, x_fake):

        #TASK CLASSIFIER
        conv_list = ()
        for layer in range(N_CONV):
            x = self.feature_extractor[layer](x)
            conv_list += x,
            
        x = conv_list[-1]
        
        h_avgpool = self.avgpool(x)
        
        x = h_avgpool
        x = x.view(x.size(0), -1)
        
        h_task = () 
        for layer in range(N_FC):
            x = self.task_classifier[layer](x)
            h_task += x,   

        #ADV CLASSIFIER
        conv_list_fake = ()
        for layer in range(N_CONV):
            x_fake = self.feature_extractor[layer](x_fake)
            conv_list_fake += x_fake,
            
        x = conv_list_fake[-1]
        
        h_avgpool = self.avgpool(x)
        
        x = h_avgpool
        x = x.view(x.size(0), -1)
        
        h_adv = () 
        for layer in range(N_FC):
            x = self.adv_classifier[layer](x)
            h_adv += x,  
            
        return (conv_list, h_task, conv_list_fake, h_adv)
    
    def predict(self, x, x_fake):
        
        #probabilities of each class
        conv_list, h_task, conv_list_fake, h_adv = self.forward(x, x_fake)
        probs = F.softmax(h_task[-1], dim=1)
      
        return probs


if __name__ == '__main__':
    
    import os
    os.getcwd()
    os.chdir("../")
    os.getcwd()
    print(os.getcwd())

    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  
        print("Running on the GPU...")
    else:
        DEVICE = torch.device("cpu")
        print("Running on the CPU...")
    print()
    

    model = CNN_REG().to(DEVICE)

    print(model)
