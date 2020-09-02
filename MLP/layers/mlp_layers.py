'''
BASED ON ANA FILIPA SEQUEIRA'S PREVIOUS WORK
'''

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

    a = BasicDenseLayer(in_features=32,
                        out_features=128,
                        bnorm=True,
                        activation='relu')
    print(a)
