import torch
import torch.nn as nn
from torch import Tensor

__negative_slope__ = 0.01


def get_activation_layer(activation):
    ''' Create activation layer '''
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(__negative_slope__)
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        errmsg = 'Invalid activation'
        raise Exception(errmsg)
        
def get_padding(kernel_size):
    ''' Compute padding '''
    if kernel_size % 2:
        padding = kernel_size // 2
    else:
        padding = (kernel_size - 1) // 2
    return padding

def BasicConvLayer(in_channels,
                   out_channels,
                   kernel_size,
                   stride,
                   bnorm=True,
                   activation='relu',
                   dropout=0.0,
                   max_pool=False):
    ''' Create a composed conv layer
    (conv - bnorm - activation - dropout) '''
    # ModuleList of layers (Conv - bnorm - activation - dropout)
    padding = get_padding(kernel_size)
    layers = nn.ModuleList([
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)])

    if bnorm:
        layers.extend([nn.BatchNorm2d(out_channels)])  # Bnorm layer

    layers.extend([get_activation_layer(activation)])  # activation layer

    if dropout > 0.0:
        layers.extend([nn.Dropout2d(dropout)])
    
    if max_pool == True:
        layers.extend([nn.MaxPool2d(kernel_size=2, stride=2)])

    # Convert to Sequential
    BasicConv2D = nn.Sequential(*(layers))

    return BasicConv2D

def BasicDenseLayer(in_features,
                    out_features,
                    bnorm=True,
                    activation='linear',
                    dropout=0.0):
    ''' Create a composed dense layer
    (Linear - bnorm - activation - dropout) '''
    # ModuleList of layers (Linear - bnorm - activation - dropout)
    layers = nn.ModuleList([
        nn.Linear(in_features, out_features)
    ])

    if bnorm:
        layers.extend([nn.BatchNorm1d(out_features)])  # bnorm layer

    if activation != 'linear':
        layers.extend([get_activation_layer(activation)])  # activation layer

    if dropout > 0.0:
        layers.extend([nn.Dropout(dropout)])

    # Convert to Sequential
    BasicDense = nn.Sequential(*(layers))

    return BasicDense


if __name__ == '__main__':

    a = BasicConvLayer(in_channels=1,
                       out_channels=64,
                       kernel_size=5,
                       stride=2,
                       bnorm=True,
                       activation='relu')
    print(a)

    a = BasicDenseLayer(in_features=32,
                        out_features=128,
                        bnorm=True,
                        activation='relu')
    print(a)