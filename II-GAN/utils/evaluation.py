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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statistics import mean
import sys
import cv2


#sys.path.insert(0, 'utils/')
sys.path.insert(0, '../data/')
sys.path.insert(0, '../models/')

from utils import _nanargmin, normal_weights_init
from data import get_data_loaders
from gan import DISCRIMINATOR, GENERATOR
from matcher import MATCHER

loss = nn.BCEWithLogitsLoss()
sigmoid = nn.Sigmoid()


def eval_model(models, data_loader, device, epoch, num_epochs, matcher_epochs, debug=False, with_generator=True):

    matcher_epochs = matcher_epochs + 1

    netD, netG = models

    with torch.no_grad():

        netD.eval()
        netG.eval()
        
        loss_c = 0
        loss_g = 0
        N = 0
        N_fake = 0
        n_correct = 0   

        m_score = 0    
        
        for i, (x,y) in enumerate(data_loader, 0):
        
            # (1) Update D network
        
            ## Train with all-real batch
        
            netD.zero_grad()
            netG.zero_grad()
        
            x = x.to(device)
            x_real = x[y == 0]
            x_fake = x[y == 1]

            if x_fake.shape[0] < 5:
              continue


        
            #b_size = real.size(0)
        
            output_real = netD(x_real)
            D_real = torch.round(sigmoid(torch.mean(output_real, dim=(1,2,3))))
            
            real_label = torch.zeros_like(output_real, device=device)
        
            errD_real = loss(output_real, real_label)

            ## Train with all-fake batch

            if with_generator:
        
                index = [i for i in range(x_fake.shape[0])]

                index_to_modify = random.sample(range(x_fake.shape[0]), x_fake.shape[0]//2)

                index_to_maintain = [i for i in index if i not in index_to_modify]

                x_fake_to_modify = x_fake[index_to_modify,...].clone().detach()

                x_fake_to_maintain = x_fake[index_to_maintain,...].clone().detach()

                x_fake_to_modify = x_fake_to_modify.to(device)

                x_fake_modified = netG(x_fake_to_modify)

                x_fake = torch.cat([x_fake_to_maintain, x_fake_modified], dim=0)

                if epoch >= num_epochs - matcher_epochs:

                    try:
                        m_score = MATCHER(x_fake_to_modify, x_fake_modified)
                    except:
                        m_score = 10.0

            output_fake = netD(x_fake.detach())

            D_fake = torch.round(sigmoid(torch.mean(output_fake, dim=(1,2,3))))
            
            fake_label = torch.ones_like(output_fake, device=device)
        
            errD_fake = loss(output_fake, fake_label)

            errD = errD_real + errD_fake

            loss_c += errD * x.shape[0]

            output_fake = netD(x_fake)
            real_label = torch.zeros_like(output_fake, device=device)

            errG = loss(output_fake, real_label)
            
            if epoch >= num_epochs - matcher_epochs:
                errG = errG + m_score

            loss_g += errG * x_fake.shape[0]

            # Compute Acc
            N += x.shape[0]
            N_fake += x_fake.shape[0]

            real_label = torch.zeros_like(D_real, device=device)
            fake_label = torch.ones_like(D_fake, device=device)

            n_correct += torch.sum(1.*(D_real == real_label)).item()
            n_correct += torch.sum(1.*(D_fake == fake_label)).item()

        loss_c = loss_c / N
        loss_g = loss_g / N_fake
        acc = n_correct / N

        return loss_c, loss_g, acc


from sklearn import metrics
import math

loss_fn = nn.BCEWithLogitsLoss()
sigmoid = nn.Sigmoid()

def test_model(model, data_loader, device, debug=False):

    print("\n")

    with torch.no_grad():

        model.eval()
        
        loss_eval = 0
        N = 0
        n_correct = 0
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        PA = 0
        BF = 0
        
        eer_list = []
        
        BPCER_APCER1_list = []
        BPCER_APCER5_list = []
        BPCER_APCER10_list = []
        
        APCER1_list = []
        APCER5_list = []
        APCER10_list = []
        
        for i, (x, y) in enumerate(data_loader):

            sys.stdout.write("\r" + 'Testing classifier... {}-th test batch'.format(i+1))

            # send mini-batch to gpu
            x = x.to(device)
            
            y = y.to(device)

            y_pred = model(x)

            y_pred = torch.mean(y_pred, dim=(1,2,3))

            # Compute cnn loss
            loss = loss_fn(y_pred, y)
            loss_eval += loss * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.round(sigmoid(y_pred))
            n_correct += torch.sum(1.*(ypred_ == y)).item()
            
            y = y.cpu().numpy()
            ypred_ = ypred_.cpu().numpy()
  
            # Biometric metrics
  
            TP += np.sum(np.logical_and(ypred_, y))
            TN += np.sum(np.logical_and(1-ypred_, 1-y))
  
            FP += np.sum(np.logical_and(ypred_, 1-y))
            FN += np.sum(np.logical_and(1-ypred_, y))
  
            PA += np.sum(y == 0)
            BF += np.sum(y == 1)
            
            probs = F.softmax(y_pred, 0)

            probs = probs.cpu().numpy()
  
            fpr, tpr, threshold = metrics.roc_curve(y, probs)
            fnr = 1 - tpr 
                       
            BPCER_APCER1_list.append(fpr[(np.abs(fnr - 0.01)).argmin()])
            BPCER_APCER5_list.append(fpr[(np.abs(fnr - 0.05)).argmin()])
            BPCER_APCER10_list.append(fpr[(np.abs(fnr - 0.1)).argmin()])
            
            APCER1_list.append(fnr[(np.abs(fnr - 0.01)).argmin()])
            APCER5_list.append(fnr[(np.abs(fnr - 0.05)).argmin()])
            APCER10_list.append(fnr[(np.abs(fnr - 0.1)).argmin()])
            
            index = _nanargmin(np.absolute((fnr - fpr)))
            if math.isnan(index) == False:
                eer_list.append(fpr[index])

        loss_eval = loss_eval / N
        acc = n_correct / N
        APCER = (FP * 1.) / (FP + TN)
        BPCER = (FN * 1.) / (FN + TP)
          
        BPCER_APCER1=mean(BPCER_APCER1_list)
        BPCER_APCER5=mean(BPCER_APCER5_list)
        BPCER_APCER10=mean(BPCER_APCER10_list)
        
        APCER1=mean(APCER1_list)
        APCER5=mean(APCER5_list)
        APCER10=mean(APCER10_list)
        
        if eer_list != []:
            EER = mean(eer_list)
        else:
            EER = -1000000000
        
        return loss_eval, acc, APCER, BPCER, EER, BPCER_APCER1, BPCER_APCER5, BPCER_APCER10, APCER1, APCER5, APCER10 
