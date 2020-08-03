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

sys.path.insert(0, 'utils/')
sys.path.insert(0, 'data/')
sys.path.insert(0, 'models/')

from utils import _nanargmin, normal_weights_init
from evaluation import eval_model, test_model
from data import get_data_loaders
from gan import DISCRIMINATOR, GENERATOR
from matcher import MATCHER

BATCH_SIZE = 32

IMG_SIZE = 128

loss = nn.BCEWithLogitsLoss()

sigmoid = nn.Sigmoid()

LEARNING_RATE = 2e-4

#-------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# Training LooP

def fit(models, dataset, fake_material, data, num_epochs, matcher_epochs, device, with_generator = True, just_train_classifier = False):

    matcher_epochs = matcher_epochs + 1

    train_loader, valid_loader = data
    netD, netG = models   

    if just_train_classifier == True:
        path =  "/ctm-hdd-pool01/afpstudents/jaf/LIVEGEN_" + dataset + "_material" + str(fake_material) + "_" + str(500) + "epochs_"
        netG.load_state_dict(torch.load(path + 'Generator.pth'))

    model_path =  "/ctm-hdd-pool01/afpstudents/jaf/LIVEGEN_" + dataset + "_material" + str(fake_material) + "_" + str(num_epochs) + "epochs_"
    output_path = f"results/{DATASET}/{DATASET}_{TOA}_material{PAI}_{EPOCHS}epochs_"

    # Start training
    train_history = {'train_c_loss': [], 'train_g_loss': [], 'train_acc': [], 'val_c_loss': [], 'val_g_loss': [], 'val_acc': []}
    
    netD.apply(normal_weights_init)

    if just_train_classifier == False:
        netG.apply(normal_weights_init)
      
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE)
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE)
    
    netD.train()
    netG.train()

    score_matcher = 0
    n_batches = 0
      
    for epoch in range(num_epochs):
      
      print("\n")
      
      g_loss = []
      d_loss = []

      d_real = []
      d_fake = []
      
      for i, (x,y) in enumerate(train_loader, 0):
      
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
          D_x = sigmoid(output_real.mean()).item()
          
          real_label = torch.zeros_like(output_real, device=device)
      
          errD_real = loss(output_real, real_label)

          ## Train with all-fake batch

          if with_generator:
      
            index = [i for i in range(x_fake.shape[0])]

            index_to_modify = random.sample(range(x_fake.shape[0]), x_fake.shape[0]//2)

            index_to_maintain = [i for i in index if i not in index_to_modify]

            x_fake_to_modify = x_fake[index_to_modify,...].clone().detach()

            x_fake_to_maintain = x_fake[index_to_maintain,...].clone().detach()

            x_fake_modified = netG(x_fake_to_modify)

            x_fake = torch.cat([x_fake_to_maintain, x_fake_modified], dim=0)

            if epoch >= num_epochs - matcher_epochs and just_train_classifier == False:

                try:
                    m_score = MATCHER(x_fake_to_modify, x_fake_modified)
                except:
                    m_score = 10.0

            if epoch == num_epochs-1:
                try:
                    score_matcher = score_matcher + MATCHER(x_fake_to_modify, x_fake_modified)
                except:
                    score_matcher = score_matcher + 10.0

                n_batches = n_batches + 1

          output_fake = netD(x_fake.detach())

          D_G_z = sigmoid(output_fake.mean()).item()
          
          fake_label = torch.ones_like(output_fake, device=device)
      
          errD_fake = loss(output_fake, fake_label)

          errD = errD_real + errD_fake

          errD.backward()
          optimizerD.step()
           
          # (2) Update G network

          if with_generator:
      
            netD.zero_grad()
            netG.zero_grad()
        
            output_fake = netD(x_fake)
            real_label = torch.zeros_like(output_fake, device=device)
            errG = loss(output_fake, real_label)

            if epoch >= num_epochs - matcher_epochs and just_train_classifier == False:
                errG = errG + m_score

            if just_train_classifier == False:
                errG.backward()
                optimizerG.step()
      
          #######################################################################
          #######################################################################
    
          sys.stdout.write("\r" + 'EPOCH [{}/{}] ..... {}-th batch: D_real = {:.3f} | D_fake = {:.3f}'.format(epoch+1, num_epochs, i+1, D_x, D_G_z))
            
      #Progress with fixed noise
      if with_generator and just_train_classifier == False:
        with torch.no_grad():    
            x_fake_modified = netG(x_fake_to_modify)
            save_images(x_fake_to_modify[:3], x_fake_modified[:3], dataset, fake_material, epoch)
      
      tr_c_loss, tr_g_loss, tr_acc = eval_model((netD, netG), train_loader, device, epoch, num_epochs, matcher_epochs, with_generator = with_generator)
      train_history['train_c_loss'].append(tr_c_loss.item())
      train_history['train_g_loss'].append(tr_g_loss.item())
      train_history['train_acc'].append(tr_acc)

      val_c_loss, val_g_loss, val_acc = eval_model((netD, netG), valid_loader, device,  epoch, num_epochs, matcher_epochs, with_generator = with_generator)
      train_history['val_c_loss'].append(val_c_loss.item())
      train_history['val_g_loss'].append(val_g_loss.item())
      train_history['val_acc'].append(val_acc)

      # display the training loss
      print()
      print( '\n>> Train: C_loss = {:.3f}  |'.format(tr_c_loss.item()) + ' G_loss = {:.3f}  |'.format(tr_g_loss.item()) + ' Acc = {:.3f}'.format(tr_acc) )
      print( '\n>> Valid: C_loss = {:.3f}  |'.format(val_c_loss.item()) + ' G_loss = {:.3f}  |'.format(val_g_loss.item()) + ' Acc = {:.3f}'.format(val_acc) )
      print()

      if epoch == num_epochs-1 and with_generator and just_train_classifier == False:
          score_matcher = score_matcher / n_batches
          print('\n>> Average matching score = {:.3f}'.format(score_matcher))

    # save train/valid history
    plot_fn = output_path + 'LIVEGEN_history.png'
    plot_train_history(train_history, plot_fn=plot_fn)

    #load last model
    torch.save(netD.state_dict(), model_path + 'Discriminator.pth')
    if just_train_classifier == False:
        torch.save(netG.state_dict(), model_path + 'Generator.pth')
      
    return (netD, train_history)


def save_images(original_images, modified_images, dataset, fake_material, epoch):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle("Epoch {}".format(epoch+1))
  ax1.axis("off")
  ax2.axis("off")
  ax1.imshow(np.transpose(vutils.make_grid(original_images, padding=2, normalize=True, nrow= 1).cpu(), (1,2,0)), vmin=-1, vmax=1)
  ax1.set_title("Original")
  ax2.imshow(np.transpose(vutils.make_grid(modified_images, padding=2, normalize=True, nrow= 1).cpu(), (1,2,0)), vmin=-1, vmax=1)
  ax2.set_title("Modified")
  fig.savefig('results/' + dataset + '/images_' + str(fake_material) + '/' + 'epoch_' + str(epoch+1) + '.png')


def plot_train_history(train_history, plot_fn):
    plt.switch_backend('agg')

    best_val_epoch = np.argmin(train_history['val_c_loss'])
    best_val_acc = train_history['val_acc'][best_val_epoch]
    best_val_c_loss = train_history['val_c_loss'][best_val_epoch]
    best_val_g_loss = train_history['val_g_loss'][best_val_epoch]
    
    plt.figure(figsize=(7, 5))
    epochs = len(train_history['train_c_loss'])
    x = range(epochs)

    plt.subplot(311)
    plt.plot(x, train_history['train_c_loss'], 'r-')
    plt.plot(x, train_history['val_c_loss'], 'g-')
    plt.plot(best_val_epoch, best_val_c_loss, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val classifier loss')
    plt.legend(['train_clf_loss', 'val_clf_loss'])
    plt.axis([0, epochs, 0, max(train_history['train_c_loss'])])

    plt.subplot(312)
    plt.plot(x, train_history['train_g_loss'], 'r-')
    plt.plot(x, train_history['val_g_loss'], 'g-')
    plt.plot(best_val_epoch, best_val_g_loss, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val generator loss')
    plt.legend(['train_gen_loss', 'val_gen_loss'])
    plt.axis([0, epochs, 0, max(train_history['train_g_loss'])])

    plt.subplot(313)
    plt.plot(x, train_history['train_acc'], 'r-')
    plt.plot(x, train_history['val_acc'], 'g-')
    plt.plot(best_val_epoch, best_val_acc, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val acc')
    plt.legend(['train_acc', 'val_acc'])
    plt.axis([0, epochs, 0, 1])

    plt.savefig(plot_fn)
    plt.close()  
    
IMG_PATH = "/ctm-hdd-pool01/DB/LivDet2015/train/"
#IMG_PATH = "L:/FPAD/Dataset/LivDet2015/train/"
#IMG_PATH = "/content/drive/My Drive/FPAD/Dataset/LivDet2015/train/"

EPOCHS = 250
BATCH_SIZE = 32
IMG_SIZE = 224

print()
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("[Device] - GPU")
else:
    DEVICE = torch.device("cpu")
    print("[Device] - CPU")
print()

#DATASET = input("Dataset [CrossMatch/Digital_Persona/GreenBit/Hi_Scan/Time_Series]: ")
#DATASET = "Digital_Persona"

UNSEEN_ATTACK = True
USE_GENERATOR = True
EPOCHS_WITH_MATCHER = 125

if USE_GENERATOR:
    TG = "wGen"
else:
    TG = "noGen"

if UNSEEN_ATTACK:
    TOA = "UA"
else:
    TOA = "OA"

#   ["CrossMatch", "Digital_Persona", "GreenBit", "Hi_Scan", "Time_Series"]

for DATASET in ["CrossMatch", "Digital_Persona", "GreenBit", "Hi_Scan", "Time_Series"]:

    if DATASET == "CrossMatch" or DATASET=="Time_Series":
        NUM_MATERIALS = 3
        TEST_MATERIALS = [0, 1, 2]
    else:
        NUM_MATERIALS = 4
        TEST_MATERIALS = [0, 1, 2, 3]

    results = []

    for PAI in TEST_MATERIALS:

        netD = DISCRIMINATOR().to(DEVICE)
        netG = GENERATOR().to(DEVICE)

        print("[Dataset] - " + DATASET + " -> Material number " + str(PAI))
        
        train_loader, valid_loader, test_loader = get_data_loaders(IMG_PATH, DATASET, test_material = PAI, img_size = IMG_SIZE, batch_size = BATCH_SIZE, croped=True, unseen_attack=UNSEEN_ATTACK)

        #netD, train_history = fit((netD, netG), DATASET, PAI, (train_loader, valid_loader), EPOCHS, EPOCHS_WITH_MATCHER, DEVICE, with_generator = USE_GENERATOR)

        netD, train_history = fit((netD, netG), DATASET, PAI, (train_loader, valid_loader), EPOCHS, EPOCHS_WITH_MATCHER, DEVICE, with_generator = USE_GENERATOR, just_train_classifier = True)

        test_loss, test_acc, test_apcer, test_bpcer, test_eer, test_bpcer_apcer1, test_bpcer_apcer5, test_bpcer_apcer10, test_apcer1, test_apcer5, test_apcer10 = test_model(netD, test_loader, DEVICE)

        results.append((test_loss.item(), test_acc, test_apcer, test_bpcer, test_eer, test_bpcer_apcer1, test_bpcer_apcer5, test_bpcer_apcer10, test_apcer1, test_apcer5, test_apcer10))

    #PRINTS -------------------------------------------------------------------------------------

    # Compute average and std
    acc_array = np.array([i[1] for i in results])
    apcer_array = np.array([i[2] for i in results])
    bpcer_array = np.array([i[3] for i in results])
    eer_array = np.array([i[4] for i in results])
    bpcer_apcer1_array = np.array([i[5] for i in results])
    bpcer_apcer5_array = np.array([i[6] for i in results])
    bpcer_apcer10_array = np.array([i[7] for i in results])
    apcer1_array = np.array([i[8] for i in results])
    apcer5_array = np.array([i[9] for i in results])
    apcer10_array = np.array([i[10] for i in results])

    print()
    print(">> TEST RESULTS [Acc, APCER, BPCER, EER, BPCER@APCER=1%, BPCER@APCER=5%, BPCER@APCER=10%, APCER1, APCER5, APCER10]:")
    print()

    results_test = []

    for j in range(len(list(acc_array))):
        res = []
        res.append(acc_array[j])
        res.append(apcer_array[j])
        res.append(bpcer_array[j])
        res.append(eer_array[j])
        res.append(bpcer_apcer1_array[j])
        res.append(bpcer_apcer5_array[j])
        res.append(bpcer_apcer10_array[j])
        res.append(apcer1_array[j])
        res.append(apcer1_array[j])
        res.append(apcer10_array[j])
                
        print(*res, sep = ", ") 
        
        results_test.append(res)

    np.savetxt(DATASET + '_' + TOA + '_' + TG + '_' + str(EPOCHS) + 'epochs_test.txt', results_test, fmt='%.3f', delimiter=',')        