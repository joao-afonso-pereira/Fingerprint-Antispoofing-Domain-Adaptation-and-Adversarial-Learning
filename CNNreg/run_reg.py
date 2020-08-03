import argparse
import os
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from statistics import mean
import pickle
import math
import sys

sys.path.insert(0, 'utils/')
sys.path.insert(0, 'data/')
sys.path.insert(0, 'models/')
sys.path.insert(0, 'layers/')
sys.path.insert(0, 'losses/')

from cnn2_utils import _nanargmin, frange
from data_cnn2 import get_data_loaders
from cnn_reg import CNN_REG
from losses_fpad import softCrossEntropyUniform, signer_transfer_loss_

#%%

# layers to regularize
CONV_LAYERS = -1

def split_batch_per_signer(x, y, g_norm, h_conv, y_task, n_signers):

    
    """split data per signer identity

    Parameters:
    x (type): batch of data
    y (type): batch of gesture labels
    g_norm (type): batch of signer iodentities labels
    h_conv (type): activations of conv layers
    h_dense (type): activations of dense layers
    y_task (type): class labels predictions
    n_signers (type): number of training signer identities

    Returns:
    x_split (type): x splitted by signer identity
    y_split (type): y splitted by signer identity
    g_split (type): g splitted by signer identity
    h_conv_split (type): h_conv splitted by signer identity
    h_dense_split (type): h_dense splitted by signer identity
    y_task_split (type): y_task splitted by signer identity

    """
    x_split = [False]*n_signers
    y_split = [False]*n_signers
    g_split = [False]*n_signers
    y_task_split = [False]*n_signers
    h_conv_split = [False]*n_signers

    for s in range(n_signers):
        x_split[s] = x[g_norm == s]
        y_split[s] = y[g_norm == s]
        g_split[s] = g_norm[g_norm == s]

        h_conv_split[s] = [torch.mean(h[g_norm == s], dim=0)
                           for h in h_conv[CONV_LAYERS:]]
        y_task_split[s] = y_task[g_norm == s]

    return x_split, y_split, g_split, h_conv_split, y_task_split

#%% 

EPOCHS = 100

loss_fn = F.cross_entropy

LEARNING_RATE = 1e-04
REG = 1e-04

ADV_WEIGHT_LIST = frange(0.1, 1, 0.1) #for optimization
TRANSFER_WEIGHT_LIST = frange(0.1, 1, 0.1) #for optimization

def fit(model, adv_weight, transfer_weight, data, n_fake, device, model_path, output):
    # train and validation loaders
    train_loader, valid_loader = data
    print("Train/Val batches: {}/{}".format(len(train_loader),
                                            len(valid_loader)))
    
    print("\n * ADV_WEIGHT = {}".format(adv_weight))
    print(" * TRANSFER_WEIGHT = {}".format(transfer_weight))

    # Set the optimizer
    task_opt = torch.optim.Adam(list(model.feature_extractor.parameters()) + 
                                list(model.task_classifier.parameters()),
                                lr=LEARNING_RATE,
                                weight_decay=REG)

    adv_opt = torch.optim.Adam(list(model.adv_classifier.parameters()),
                               lr=LEARNING_RATE,
                               weight_decay=REG)

    # Start training
    train_history = {'train_loss': [], 'train_acc': [], 'train_apcer': [], 'train_bpcer': [], 'train_eer': [], 'train_bpcer_apcer1': [], 'train_bpcer_apcer5': [], 'train_bpcer_apcer10': [], 'train_apcer1': [], 'train_apcer5': [], 'train_apcer10': [],
                     'val_loss': [], 'val_acc': [], 'val_apcer': [], 'val_bpcer': [], 'val_eer': [], 'val_bpcer_apcer1': [], 'val_bpcer_apcer5': [], 'val_bpcer_apcer10': [], 'val_apcer1': [], 'val_apcer5': [], 'val_apcer10': []}

    # Best validation params
    best_val = -float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        print('\nEPOCH {}/{}\n'.format(epoch + 1, EPOCHS))

        # TRAINING
        # set model to train
        model.train()
        for i, (x, y, f, f_norm) in enumerate(train_loader): 

            x = x.to(device)
            x_fake = x[f != -1]
            
            if x.shape[0] == 0 or x_fake.shape[0] == 0:
                continue
            
            y = y.type(torch.LongTensor).to(device)
            f_norm_fake = f_norm[f != -1].type(torch.LongTensor).to(device)
            
            # forward pass
            conv_list, h_task, conv_list_fake, h_adv = model(x, x_fake)
            y_task = h_task[-1]
            y_adv = h_adv[-1]

            # Compute vae loss
            task_loss = loss_fn(y_task, y)
            adv_loss = loss_fn(y_adv, f_norm_fake)
            
            x_split, y_split, g_split, h_conv_split, y_task_split = split_batch_per_signer(x, y, f_norm, conv_list, y_task, n_fake-1)
            fake_on_batch = [i for i in range(len(g_split)) if len(g_split[i])]

            if len(fake_on_batch) <= 1:
                transfer_loss = torch.tensor(0.0)
            else:
                transfer_loss = signer_transfer_loss_(h_conv_split, fake_on_batch)
            
            loss = task_loss + adv_weight*softCrossEntropyUniform(y_adv) + transfer_weight*transfer_loss

            task_opt.zero_grad()
            loss.backward(retain_graph=True)
            task_opt.step()

            adv_opt.zero_grad()
            adv_loss.backward()
            adv_opt.step()

            # display the mini-batch loss
            sys.stdout.write("\r" + '........{}-th mini-batch loss: {:.3f} | task_loss: {:.3f} | CEU_loss: {:.3f} | adv_loss: {:.3f} | transfer_loss: {:.3f} '.format(i+1, loss.item(), task_loss.item(), softCrossEntropyUniform(y_adv).item(), adv_loss.item(), transfer_loss.item()))
            sys.stdout.flush()
            
        # Validation
        tr_loss, tr_acc, tr_apcer, tr_bpcer, tr_eer, tr_bpcer_apcer1, tr_bpcer_apcer5, tr_bpcer_apcer10, tr_apcer1, tr_apcer5, tr_apcer10 = eval_model(model, train_loader, n_fake, adv_weight, transfer_weight, device)
        train_history['train_loss'].append(tr_loss.item())
        train_history['train_acc'].append(tr_acc)
        train_history['train_apcer'].append(tr_apcer)
        train_history['train_bpcer'].append(tr_bpcer)
        train_history['train_eer'].append(tr_eer)
        train_history['train_bpcer_apcer1'].append(tr_bpcer_apcer1)
        train_history['train_bpcer_apcer5'].append(tr_bpcer_apcer5)
        train_history['train_bpcer_apcer10'].append(tr_bpcer_apcer10)
        train_history['train_apcer1'].append(tr_apcer1)
        train_history['train_apcer5'].append(tr_apcer5)
        train_history['train_apcer10'].append(tr_apcer10)

        val_loss, val_acc, val_apcer, val_bpcer, val_eer, val_bpcer_apcer1, val_bpcer_apcer5, val_bpcer_apcer10, val_apcer1, val_apcer5, val_apcer10 = eval_model(model, valid_loader, n_fake, adv_weight, transfer_weight, device)
        train_history['val_loss'].append(val_loss.item())
        train_history['val_acc'].append(val_acc)
        train_history['val_apcer'].append(val_apcer)
        train_history['val_bpcer'].append(val_bpcer)
        train_history['val_eer'].append(val_eer)
        train_history['val_bpcer_apcer1'].append(val_bpcer_apcer1)
        train_history['val_bpcer_apcer5'].append(val_bpcer_apcer5)
        train_history['val_bpcer_apcer10'].append(val_bpcer_apcer10)
        train_history['val_apcer1'].append(val_apcer1)
        train_history['val_apcer5'].append(val_apcer5)
        train_history['val_apcer10'].append(val_apcer10)


        # save best validation model
        if best_val < val_acc:
            torch.save(model.state_dict(), model_path + 'cnn2_fpad.pth')
            best_val = val_acc
            best_epoch = epoch

        # display the training loss
        print()
        print('\n>> Train loss: {:.3f}  |'.format(tr_loss.item()) + ' Train Acc: {:.3f}'.format(tr_acc) + '\n   Train APCER: {:.3f} |'.format(tr_apcer) + ' Train BPCER: {:.3f}'.format(tr_bpcer) + '\n   Train EER: {:.3f}'.format(tr_eer))

        print('\n>> Valid loss: {:.3f}  |'.format(val_loss.item()) + ' Valid Acc: {:.3f}'.format(val_acc) + '\n   Valid APCER: {:.3f} |'.format(val_apcer) + ' Valid BPCER: {:.3f}'.format(val_bpcer) + '\n   Valid EER: {:.3f}'.format(val_eer))

        print('\n>> Best model: {} / Acc={:.3f}'.format(best_epoch+1, best_val))
        print()

    # save train/valid history
    plot_fn = output + 'cnn2_fpad_history.png'
    plot_train_history(train_history, plot_fn=plot_fn)

    # return best validation model
    model.load_state_dict(torch.load(model_path + 'cnn2_fpad.pth'))

    return model, train_history, valid_loader, best_epoch+1


def plot_train_history(train_history, plot_fn=None):
    plt.switch_backend('agg')

    best_val_epoch = np.argmin(train_history['val_loss'])
    best_val_acc = train_history['val_acc'][best_val_epoch]
    best_val_loss = train_history['val_loss'][best_val_epoch]
    plt.figure(figsize=(7, 5))
    epochs = len(train_history['train_loss'])
    x = range(epochs)
    plt.subplot(211)
    plt.plot(x, train_history['train_loss'], 'r-')
    plt.plot(x, train_history['val_loss'], 'g-')
    plt.plot(best_val_epoch, best_val_loss, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val loss')
    plt.legend(['train_loss', 'val_loss'])
    plt.axis([0, epochs, 0, max(train_history['train_loss'])])
    plt.subplot(212)
    plt.plot(x, train_history['train_acc'], 'r-')
    plt.plot(x, train_history['val_acc'], 'g-')
    plt.plot(best_val_epoch, best_val_acc, 'bx')
    plt.xlabel('Epoch')
    plt.ylabel('Train/Val acc')
    plt.legend(['train_acc', 'val_acc'])
    plt.axis([0, epochs, 0, 1])
    if plot_fn:
        #plt.show()
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()


def eval_model(model, data_loader, n_fake, adv_weight, transfer_weight, device, debug=False):
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
        
        for i, (x, y, f, f_norm) in enumerate(data_loader):
            
            # send mini-batch to gpu
            x = x.to(device)
            x_fake = x[f != -1]
            
            if x.shape[0] == 0 or x_fake.shape[0] == 0:
                continue
            
            y = y.type(torch.LongTensor).to(device)
            f_norm_fake = f_norm[f != -1].type(torch.LongTensor).to(device)

            # forward pass
            conv_list, h_task, conv_list_fake, h_adv = model(x, x_fake)           
            y_task = h_task[-1]
            y_adv = h_adv[-1]

            # Compute vae loss
            task_loss = loss_fn(y_task, y)
            adv_loss = loss_fn(y_adv, f_norm_fake)
            
            x_split, y_split, g_split, h_conv_split, y_task_split = split_batch_per_signer(x, y, f_norm, conv_list, y_task, n_fake-1)
            fake_on_batch = [i for i in range(len(g_split)) if len(g_split[i])]

            if len(fake_on_batch) <= 1:
                transfer_loss = torch.tensor(0.0)
            else:
                transfer_loss = signer_transfer_loss_(h_conv_split, fake_on_batch)
                
            loss = task_loss + adv_weight*softCrossEntropyUniform(y_adv) + transfer_weight*transfer_loss

            # Compute cnn loss
            loss_eval += loss * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.argmax(y_task, dim=1)
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
            
            probs = model.predict(x, x_fake)
            
            probs = probs.cpu().numpy()

            probs = probs[:, 1]
  
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
            EER = -float('inf')
        
        return loss_eval, acc, APCER, BPCER, EER, BPCER_APCER1, BPCER_APCER5, BPCER_APCER10, APCER1, APCER5, APCER10 


def main():
    
    IMG_PATH = "/ctm-hdd-pool01/DB/LivDet2015/train/"
    #IMG_PATH = "L:/FPAD/Dataset/LivDet2015/train/"
    
    CUDA = 0
    
    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:" + str(CUDA))  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("[Device] - GPU")
    else:
        DEVICE = torch.device("cpu")
        print("[Device] - CPU")
    
    mode = input("Enter the mode [train/optim/test]: ") #train
    data_ = input("Dataset [ALL/CrossMatch/Digital_Persona/GreenBit/Hi_Scan/Time_Series]: ")
        
    if mode=="optim":
        unseen_ = "y"
    else:
        unseen_ = input("Unseen attack? [y/n]: ")        
    
    if unseen_ == "y":
        unseen = True
        NUM_ITERATIONS = 1
        attack_txt = "UA"
    elif unseen_ == "n":
        unseen = False 
        NUM_ITERATIONS = 3
        attack_txt = "OA"
    else:
        sys.exit("Error ('Unseen attack?'): incorrect input!")
        
    if data_ == "ALL":    
        sensors = ["CrossMatch", "Digital_Persona", "GreenBit", "Hi_Scan", "Time_Series"]    
    else:    
        sensors = [data_]     

    for DATASET in sensors:
        
        print("\n[Dataset] - " + DATASET + "\n")
    
        
        if DATASET == "CrossMatch" or DATASET=="Time_Series":
            NUM_MATERIALS = 3
        else:
            NUM_MATERIALS = 4
            
        # For LOOP - Test splits
        train_results_ = []
        results = []
        best_epochs = []   
        optimization = []
        
        for iteration in range(NUM_ITERATIONS):
            
            print("\n-- ITERATION {}/{} --".format(iteration+1, NUM_ITERATIONS))
    
            for test_material in range(NUM_MATERIALS):
                
                output_fn = "results/" + DATASET + "/" + DATASET + "_" + str(test_material) + "_"
                model_path = "/ctm-hdd-pool01/afpstudents/jaf/CNN2_" + DATASET + "_" + str(test_material) + "_"
                
                n_fake_train = -1
                if unseen_ == "y":
                    n_fake_train = NUM_MATERIALS-1
                if unseen_ == "n":
                    n_fake_train = 1
                
                # Train, optimize or test
                if mode == 'train':

                    model = CNN_REG().to(DEVICE)
                    
                    ADV_WEIGHT = 0.47    #0.4 #0.47
                    TRANSFER_WEIGHT = 0.51 #0.11 #0.51
                
                    (train_loader, valid_loader, test_loader) = get_data_loaders(IMG_PATH, DATASET, test_material, croped=True, unseen_attack=unseen)
                    
                    
                    # Fit model
                    model, train_history, _, best_epoch = fit(model=model,
                                                              adv_weight = ADV_WEIGHT,
                                                              transfer_weight = TRANSFER_WEIGHT,
                                                              data=(train_loader, valid_loader),
                                                              n_fake = NUM_MATERIALS,
                                                              device=DEVICE,
                                                              model_path = model_path,                                 
                                                              output=output_fn)
            
                    # save train history
                    train_res_fn = output_fn + "history.pckl"
                    pickle.dump(train_history, open(train_res_fn, "wb"))
                    
                    #Train results
                    train_results = pickle.load(open(train_res_fn, "rb"))
                    train_results_.append([train_results['train_acc'][EPOCHS-1], train_results['train_apcer'][EPOCHS-1], train_results['train_bpcer'][EPOCHS-1], train_results['train_eer'][EPOCHS-1], train_results['train_bpcer_apcer1'][EPOCHS-1], train_results['train_bpcer_apcer5'][EPOCHS-1], train_results['train_bpcer_apcer10'][EPOCHS-1], train_results['train_apcer1'][EPOCHS-1], train_results['train_apcer5'][EPOCHS-1], train_results['train_apcer10'][EPOCHS-1]])
                    
                    # Test results
                    test_loss, test_acc, test_apcer, test_bpcer, test_eer, test_bpcer_apcer1, test_bpcer_apcer5, test_bpcer_apcer10, test_apcer1, test_apcer5, test_apcer10 = eval_model(model, test_loader, NUM_MATERIALS, ADV_WEIGHT, TRANSFER_WEIGHT, DEVICE)
                    print('\nTest loss: {:.3f}            |'.format(test_loss.item()) + ' Test Acc: {:.3f}'.format(test_acc) + '\nTest APCER: {:.3f}           |'.format(test_apcer) + ' Test BPCER: {:.3f}'.format(test_bpcer))     
                    print('Test BPCER@APCER=1%: {:.3f}  | Test APCER1: {:.3f}'.format(test_bpcer_apcer1, test_apcer1))
                    print('Test BPCER@APCER=5%: {:.3f}  | Test APCER5: {:.3f}'.format(test_bpcer_apcer5, test_apcer5))
                    print('Test BPCER@APCER=10%: {:.3f} | Test APCER10: {:.3f}'.format(test_bpcer_apcer10, test_apcer10))
                    print('Test EER: {:.3f}'.format(test_eer))
                    results.append((test_loss.item(), test_acc, test_apcer, test_bpcer, test_eer, test_bpcer_apcer1, test_bpcer_apcer5, test_bpcer_apcer10, test_apcer1, test_apcer5, test_apcer10))
                    
                    best_epochs.append(best_epoch)
                
                    # save results
                    res_fn = output_fn + 'results.pckl'
                    pickle.dump(results, open(res_fn, "wb"))
                    results = pickle.load(open(res_fn, "rb"))                    
                    
                elif mode == 'optim':
                    
                    best_accuracy = -float('inf')
                    best_adv_weight = -float('inf')
                    best_epoch_ = -float('inf')
                    
                    step = 0

                    #for ADV_WEIGHT in ADV_WEIGHT_LIST:  

                    ADV_WEIGHT = 0
                        
                    for TRANSFER_WEIGHT in TRANSFER_WEIGHT_LIST:

                        model = CNN_REG().to(DEVICE)
                        
                        step = step + 1
                
                        (train_loader, valid_loader, test_loader) = get_data_loaders(IMG_PATH, DATASET, test_material, croped=True, unseen_attack=unseen)
                        
                        
                        # Fit model
                        model, train_history, _, best_epoch = fit(model=model,
                                                                    adv_weight = ADV_WEIGHT,       
                                                                    transfer_weight = TRANSFER_WEIGHT,
                                                                    data=(train_loader, valid_loader),
                                                                    n_fake = NUM_MATERIALS,
                                                                    device=DEVICE,
                                                                    model_path = model_path,                                 
                                                                    output=output_fn)
                        
                        # save train history
                        train_res_fn = output_fn + "history_reg.pckl"
                        pickle.dump(train_history, open(train_res_fn, "wb"))
                        
                        #Train results
                        train_results = pickle.load(open(train_res_fn, "rb"))
                        history = [train_results['train_acc'][EPOCHS-1], train_results['train_apcer'][EPOCHS-1], train_results['train_bpcer'][EPOCHS-1], train_results['train_eer'][EPOCHS-1], train_results['train_bpcer_apcer1'][EPOCHS-1], train_results['train_bpcer_apcer5'][EPOCHS-1], train_results['train_bpcer_apcer10'][EPOCHS-1], train_results['train_apcer1'][EPOCHS-1], train_results['train_apcer5'][EPOCHS-1], train_results['train_apcer10'][EPOCHS-1]]
                        
                        test_loss, test_acc, test_apcer, test_bpcer, test_eer, test_bpcer_apcer1, test_bpcer_apcer5, test_bpcer_apcer10, test_apcer1, test_apcer5, test_apcer10 = eval_model(model, test_loader, NUM_MATERIALS-1, ADV_WEIGHT, TRANSFER_WEIGHT, DEVICE)
                        
                        if test_acc > best_accuracy:
                            best_accuracy = test_acc
                            best_adv_weight = ADV_WEIGHT
                            best_transfer_weight = ADV_WEIGHT
                            best_epoch_ = best_epoch
                            test_results = (test_loss.item(), test_acc, test_apcer, test_bpcer, test_eer, test_bpcer_apcer1, test_bpcer_apcer5, test_bpcer_apcer10, test_apcer1, test_apcer5, test_apcer10)
                            tr_results = history
                    
                    optimization.append((best_accuracy, best_adv_weight, best_transfer_weight))
                            
                    train_results_.append(tr_results)
                    best_epochs.append(best_epoch_)
                    results.append(test_results)                         
            
                elif mode == 'test':
                    sys.exit("Error: in construction yet!")
                else:
                    sys.exit("Error: incorrect mode!")
            
                
            ### PRINT RESULTS -----------------------------------------------------------------------------------------------------------------------------------
            print('\n\n\n-------------------------------------------\n-------------- R E S U L T S --------------\n-------------------------------------------')        
            
            print()
            print("***************")            
            print(DATASET)            
            print("***************")
            
            optim_res = []
            
            if optimization != []:
                
                print()
                for m in range(len(optimization)):
                    print("\n>> OPTIMIZATION RESULT MATERIAL {}:\n".format(m+1))
                    print("      - Best Accuracy = {}".format(optimization[m][0]))
                    print("      - Best Adv Weight = {}".format(optimization[m][1]))
                    print("      - Best Transfer Weight = {}".format(optimization[m][2]))
                    optim_res.append(optimization[m][0])
                    optim_res.append(optimization[m][1])
                print()  
                
                optim_res = np.array(optim_res)
                np.savetxt(DATASET + '_optim.txt', optim_res, fmt='%.3f', delimiter=',')
            
                print('\n-------------------------------------------')               
            
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
            
            #Best epochs
            print('\nBest epochs:', end=" ")
            for epoch in best_epochs:
                print(epoch, end="   ")
                
            #Results of all loops (train and test)
            #Results of all loops (train and test)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print()
            print("\n[Acc, APCER, BPCER, EER, BPCER@APCER=1%, BPCER@APCER=5%, BPCER@APCER=10%, APCER1, APCER5, APCER10]")
            print()
            print(">> TRAIN RESULTS:")
            print()
            for k in range(NUM_MATERIALS):
                print(*train_results_[k], sep = ", ") 
            
            print()
            print(">> TEST RESULTS:")
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
                
            if iteration == NUM_ITERATIONS-1:        
                np.savetxt(DATASET + '_' + attack_txt + '_test.txt', results_test, fmt='%.3f', delimiter=',')

    
    print("\n\nDONE!")

if __name__ == '__main__':
    main()
