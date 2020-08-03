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

from cnn2_utils import _nanargmin
from data_cnn2 import get_data_loaders
from cnn2 import CNN2_FPAD
from cnn2_reg import CNN2_REG_FPAD


#%% 

EPOCHS = 50

loss_fn = F.cross_entropy

LEARNING_RATE = 1e-04
REG = 1e-04

def fit(model, data, device, model_path, output):
    # train and validation loaders
    train_loader, valid_loader = data
    print("Train/Val batches: {}/{}".format(len(train_loader),
                                            len(valid_loader)))

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
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
        for i, (x, y, f, _) in enumerate(train_loader):  # iterations loop
            # send mini-batch to gpu
            x = x.to(device)
        
            x_real = x[f == -1]
            x_fake = x[f != -1]
            
            y = y.type(torch.LongTensor)
            y = y.to(device)
            
            f = f.type(torch.LongTensor)
            f = f.to(device)

            # forward pass
            conv_list, h_avgpool, h_list = model(x)
            y_pred = h_list[-1]
            
            ypred_ = torch.argmax(y_pred, dim=1)
            
            # Compute loss
            loss = loss_fn(y_pred, y)

            # Backprop and optimize
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute new gradients
            optimizer.step()       # optimize the parameters

            # display the mini-batch loss
            sys.stdout.write("\r" + '........{}-th mini-batch loss: {:.3f}'.format(i, loss.item()))
            sys.stdout.flush()
            
        # Validation
        tr_loss, tr_acc, tr_apcer, tr_bpcer, tr_eer, tr_bpcer_apcer1, tr_bpcer_apcer5, tr_bpcer_apcer10, tr_apcer1, tr_apcer5, tr_apcer10 = eval_model(model, train_loader, device)
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

        val_loss, val_acc, val_apcer, val_bpcer, val_eer, val_bpcer_apcer1, val_bpcer_apcer5, val_bpcer_apcer10, val_apcer1, val_apcer5, val_apcer10 = eval_model(model, valid_loader, device)
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
        plt.show()
        plt.savefig(plot_fn)
        plt.close()
    else:
        plt.show()


def eval_model(model, data_loader, device, debug=False):
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
        
        for i, (x, y, f, _) in enumerate(data_loader):
            # send mini-batch to gpu
            x = x.to(device)
            
            x_real = x[f == -1]
            x_fake = x[f != -1]
            
            y = y.type(torch.LongTensor)
            y = y.to(device)
            
            f = f.type(torch.LongTensor)
            f = f.to(device)

            # forward pass
            conv_list, h_avgpool, h_list = model(x)           
            y_pred = h_list[-1]

            # Compute cnn loss
            loss = loss_fn(y_pred, y)
            loss_eval += loss * x.shape[0]

            # Compute Acc
            N += x.shape[0]
            ypred_ = torch.argmax(y_pred, dim=1)
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
            
            probs = model.predict(x)
            
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
            EER = -1000000000
        
        return loss_eval, acc, APCER, BPCER, EER, BPCER_APCER1, BPCER_APCER5, BPCER_APCER10, APCER1, APCER5, APCER10 


def main():
    
    IMG_PATH = "/ctm-hdd-pool01/DB/LivDet2015/train/"
    
    CUDA = 0
    
    print()
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:" + str(CUDA))
        print("Running on the GPU")
    else:
        DEVICE = torch.device("cpu")
        print("Running on the CPU")
    
    mode = input("Enter the mode [train/test]: ")
    data_ = input("Dataset [ALL/CrossMatch/Digital_Persona/GreenBit/Hi_Scan/Time_Series]: ")
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
        
        print("\nRunning in " + DATASET + "...\n")
    
        
        if DATASET == "CrossMatch" or DATASET=="Time_Series":
            NUM_MATERIALS = 3
        else:
            NUM_MATERIALS = 4
            
        # For LOOP - Test splits
        train_results_ = []
        results = []
        best_epochs = []    
        
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
                
                model = CNN2_FPAD().to(DEVICE)
                
                # Train or test
                if mode == 'train':
                    
                    (train_loader, valid_loader, test_loader) = get_data_loaders(IMG_PATH, DATASET, test_material, croped=True, unseen_attack=unseen)
                    
                    
                    # Fit model
                    model, train_history, _, best_epoch = fit(model=model,
                                                  data=(train_loader, valid_loader),
                                                  device=DEVICE,
                                                  model_path = model_path,                                 
                                                  output=output_fn)
            
                    # save train history
                    train_res_fn = output_fn + "history.pckl"
                    pickle.dump(train_history, open(train_res_fn, "wb"))
            
                elif mode == 'test':
                    sys.exit("Error: in construction yet!")
                else:
                    sys.exit("Error: incorrect mode!")
                
                #Train results
                train_results = pickle.load(open(train_res_fn, "rb"))
                train_results_.append([train_results['train_acc'][EPOCHS-1], train_results['train_apcer'][EPOCHS-1], train_results['train_bpcer'][EPOCHS-1], train_results['train_eer'][EPOCHS-1], train_results['train_bpcer_apcer1'][EPOCHS-1], train_results['train_bpcer_apcer5'][EPOCHS-1], train_results['train_bpcer_apcer10'][EPOCHS-1], train_results['train_apcer1'][EPOCHS-1], train_results['train_apcer5'][EPOCHS-1], train_results['train_apcer10'][EPOCHS-1]])
                
                # Test results
                test_loss, test_acc, test_apcer, test_bpcer, test_eer, test_bpcer_apcer1, test_bpcer_apcer5, test_bpcer_apcer10, test_apcer1, test_apcer5, test_apcer10 = eval_model(model, test_loader, DEVICE)
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
            
            print('\n\n\n---------------------------------\n-------------------------------------------')
            print('Average Acc: {:.3f}             |'.format(np.mean(acc_array)) + ' Std: {:.3f}'.format(np.std(acc_array)))
            print('Average APCER: {:.3f}           |'.format(np.mean(apcer_array)) + ' Std: {:.3f}'.format(np.std(apcer_array)))
            print('Average BPCER: {:.3f}           |'.format(np.mean(bpcer_array)) + ' Std: {:.3f}'.format(np.std(bpcer_array)))
            print('Average EER: {:.3f}             |'.format(np.mean(eer_array)) + ' Std: {:.3f}'.format(np.std(eer_array)))
            print('Average BPCER@APCER=1%: {:.3f}  |'.format(np.mean(bpcer_apcer1_array)) + ' Std: {:.3f}'.format(np.std(bpcer_apcer1_array)))
            print('Average BPCER@APCER=5%: {:.3f}  |'.format(np.mean(bpcer_apcer5_array)) + ' Std: {:.3f}'.format(np.std(bpcer_apcer5_array)))
            print('Average BPCER@APCER=10%: {:.3f} |'.format(np.mean(bpcer_apcer10_array)) + ' Std: {:.3f}'.format(np.std(bpcer_apcer10_array)))
            print('Average APCER1: {:.3f}          |'.format(np.mean(apcer1_array)) + ' Std: {:.3f}'.format(np.std(apcer1_array)))
            print('Average APCER5: {:.3f}          |'.format(np.mean(apcer5_array)) + ' Std: {:.3f}'.format(np.std(apcer5_array)))
            print('Average APCER10: {:.3f}         |'.format(np.mean(apcer10_array)) + ' Std: {:.3f}'.format(np.std(apcer10_array)))
            
            #Best epochs
            print('\nBest epochs:', end=" ")
            for epoch in best_epochs:
                print(epoch, end="   ")
                
            #Results of all loops (train and test)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print()
            print()
            print(DATASET + "results:")
            print()
            print(">>Train results [Acc, APCER, BPCER, EER, BPCER@APCER=1%, BPCER@APCER=5%, BPCER@APCER=10%, APCER1, APCER5, APCER10]")
            print()
            for k in range(NUM_MATERIALS):
                print(*train_results_[k], sep = ", ") 
            
            print()
            print()
            print(">>Test results")
            
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
