from PIL import Image
import glob
from skimage import color
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score
    
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def fpad_iso_metrics(ground_truth, prediction, probs):
    
    P = np.sum(ground_truth)
    N = len(ground_truth) - np.sum(ground_truth)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(prediction)): 
        if ground_truth[i]==prediction[i]==1:
           TP += 1
        if prediction[i]==1 and ground_truth[i]!=prediction[i]:
           FP += 1
        if ground_truth[i]==prediction[i]==0:
           TN += 1
        if prediction[i]==0 and ground_truth[i]!=prediction[i]:
           FN += 1
    
    APCER = FN/P #FALSE NEGATIVE RATE (fnr at a defined treshold)
    BPCER = FP/N #FALSE POSITIVE RATE (fpr at a defined treshold)
    
    fpr, tpr, thresholds = roc_curve(ground_truth, probs) 
    fnr = 1 - tpr
    
    BPCER25=fpr[(np.abs(fnr - 0.04)).argmin()] #closest to 4
            
    DETC = [fpr, fnr, tpr]    
    
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    AUROC = roc_auc_score(ground_truth, probs)
      
    return(APCER, BPCER, BPCER25, DETC, EER, AUROC)
    
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def normalize(x, minimum=-1, maximum=-1):
    min_max = []
    if minimum == -1 or maximum == -1:
        from sklearn.preprocessing import normalize
        norm_x = normalize(x[:,np.newaxis], axis=0).ravel()
        min_max.append(min(norm_x))
        min_max.append(max(norm_x))
        return np.array(min_max), norm_x
    else:
        norm_x=[]
        for i in range(len(x)):
            norm_x.append((x[i]-min(x))/(max(x)-min(x)) * (maximum-minimum) + minimum)
        return np.array(norm_x)

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def frange(start, stop=None, step=None):

    num = start
    _list = []
    while num <= stop:
        _list.append(num)
        num = num + step
    
    return _list

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def unseenValidation(DATASET, croped=False):
    
    PATH = 'C:/Users/Asus/Desktop/5_Ano/5_FPAD/Dataset/'
  
    if DATASET=='CrossMatch':
        files = '/*.bmp'
        percentage1 = 0.33
        percentage2 = 0.5
        materials = ['Body_Double', 'Ecoflex', 'Playdoh']
    elif DATASET=='Digital_Persona':
        files = '/*.png'
        percentage1 = 0.25
        percentage2 = 0.33
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    elif DATASET=='GreenBit':
        files = '/*.png'
        percentage1 = 0.25
        percentage2 = 0.33
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    elif DATASET=='Hi_Scan':
        files = '/*.bmp'
        percentage1 = 0.25
        percentage2 = 0.33
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    else: #Time_Series
        files = '/*.bmp'
        percentage1 = 0.33
        percentage2 = 0.5
        materials = ['Body_Double', 'Ecoflex', 'Playdoh']
    
    if croped == False:
        print("\nRunning unseen-validation on {}...\n".format(DATASET))
        live_path = '/Live'
        fake_path = '/Fake/'
        seg = 'Without_segmentation/'
        #txt = DATASET + '_withoutSegmentation_Int.txt'
        txt = DATASET + '_withoutSegmentation_LBP.txt'
        #txt = DATASET + '_withoutSegmentation_IntLBP.txt'
    else:
        files = '/*.png'
        print("\nRunning unseen-validation with segmentation on {}...\n".format(DATASET))
        live_path = '/Live_c'
        fake_path = '/Fake_c/'
        seg = 'Segmentation/'
        #txt = DATASET + '_withSegmentation_Int.txt'
        txt = DATASET + '_withSegmentation_LBP.txt'
        #txt = DATASET + '_withSegmentation_IntLBP.txt'
    
    num_clf = 3 
    num_materials = (len(materials))
    
    
    radius = 1
    n_points = 8 * radius
    METHOD = 'uniform' 
    
    #print("Processing live images...\n") 
    
    intensity_feature = []
    LBP_feature = []
    label = []
    
    for filename in glob.glob(PATH + 'LivDet2015/train/' + DATASET + live_path + files):
        with Image.open(filename) as img:
            if DATASET=='Digital_Persona':
                img = color.rgb2gray(np.array(img))
            else:
                img = np.array(img)
            hist1, bin_edges1 = np.histogram(img, density=True)
            intensity_feature.append(hist1.tolist())
            lbp = local_binary_pattern(img, n_points, radius, METHOD)
            hist2, bin_edges2 = np.histogram(lbp, density=True)
            LBP_feature.append(hist2.tolist())
            label.append(0)
        

    intensity_feature = np.array(intensity_feature)
    LBP_feature = np.array(LBP_feature)
    label = np.array(label)
      
    #X = intensity_feature
    X = LBP_feature
    #X = np.hstack((intensity_feature, LBP_feature))

    y = label
    
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=percentage1, random_state=42)
    
    X_train, X_otim, y_train, y_otim = train_test_split(X_, y_, test_size=percentage2, random_state=42)
    
    APCER_list = []
    BPCER_list = []
    BPCER25_list = []
    DETC = []
    EER_list = []
    AUROC_list = []
    
    # TEST/TRAIN SPOOF + CLASSIFICATION + METRICS ------------------------------------------------------------
    
    
    DETC_svm_otim = []
    DETC_nb_otim = []
    DETC_knn_otim = []
    
    for unseen_otimization in range(0, len(materials)):
        
        #print("\n*UNSEEN VALIDATION NUMBER {} OF {}*\n".format(unseen_otimization+1, len(materials)))
        
        APCER = np.ones((num_clf, num_materials))
        BPCER = np.ones((num_clf, num_materials))
        BPCER25 = np.ones((num_clf, num_materials))
        EER = np.ones((num_clf, num_materials))
        AUROC = np.ones((num_clf, num_materials))
        
        DETC_svm_attack = []
        DETC_nb_attack = []
        DETC_knn_attack = []
        
        for unseen_test in range(0, len(materials)):
            
            if unseen_test != unseen_otimization:
            
                intensity_feature = []
                LBP_feature = []
                label = []
                
                intensity_otimization = []
                LBP_otimization = []
                label_otimization = []
                
                intensity_test = []
                LBP_test = []
                label_test = []
                
                for k in range(0, len(materials)): 
                    
                    material=materials[k]
                    
                    if k==unseen_test:
                        
                        
                        for filename in glob.glob(PATH + 'LivDet2015/train/' + DATASET + fake_path + material + files):
                            with Image.open(filename) as img:
                                if DATASET=='Digital_Persona':
                                    img = color.rgb2gray(np.array(img))
                                else:
                                    img = np.array(img)
                                hist1, bin_edges1 = np.histogram(img, density=True)
                                intensity_test.append(hist1.tolist())
                                lbp = local_binary_pattern(img, n_points, radius, METHOD)
                                hist2, bin_edges2 = np.histogram(lbp, density=True)
                                LBP_test.append(hist2.tolist())
                                label_test.append(1)
                        
                        intensity_test = np.array(intensity_test)
                        LBP_test = np.array(LBP_test)
                        label_test = np.array(label_test)  
                            
                    elif k==unseen_otimization:
                        
                        
                        for filename in glob.glob(PATH + 'LivDet2015/train/' + DATASET + fake_path + material + files):
                            with Image.open(filename) as img:
                                if DATASET=='Digital_Persona':
                                    img = color.rgb2gray(np.array(img))
                                else:
                                    img = np.array(img)
                                hist1, bin_edges1 = np.histogram(img, density=True)
                                intensity_otimization.append(hist1.tolist())
                                lbp = local_binary_pattern(img, n_points, radius, METHOD)
                                hist2, bin_edges2 = np.histogram(lbp, density=True)
                                LBP_otimization.append(hist2.tolist())
                                label_otimization.append(1)
                        
                        intensity_otimization = np.array(intensity_otimization)
                        LBP_otimization = np.array(LBP_otimization)
                        label_otimization = np.array(label_otimization)  
                    else:
                            
                        
                        for filename in glob.glob(PATH + 'LivDet2015/train/' + DATASET + fake_path + material + files):
                            with Image.open(filename) as img:
                                if DATASET=='Digital_Persona':
                                    img = color.rgb2gray(np.array(img))
                                else:
                                    img = np.array(img)
                                hist1, bin_edges1 = np.histogram(img, density=True)
                                intensity_feature.append(hist1.tolist())
                                lbp = local_binary_pattern(img, n_points, radius, METHOD)
                                hist2, bin_edges2 = np.histogram(lbp, density=True)
                                LBP_feature.append(hist2.tolist())
                                label.append(1)
                        
                intensity_feature = np.array(intensity_feature)
                LBP_feature = np.array(LBP_feature)
                label = np.array(label)  
                
                #print("------------------------------")
                #print("-> Concluding data preparation...\n")
                
                ##################################################
                #X = intensity_feature
                X = LBP_feature
                #X = np.hstack((intensity_feature, LBP_feature))
                X_train = np.vstack((X_train, X))
            
                
                y = label
                y_train = np.array(y_train.tolist() + y.tolist())
                
                ##################################################
                #X = intensity_otimization
                X = LBP_otimization
                #X = np.hstack((intensity_otimization, LBP_otimization))
                X_otim = np.vstack((X_otim, X))
                
                y = label_otimization
                y_otim = np.array(y_otim.tolist() + y.tolist())   
                
                ##################################################
                #X = intensity_test
                X = LBP_test
                #X = np.hstack((intensity_test, LBP_test))
                X_test = np.vstack((X_test, X))
                
                y = label_test
                y_test = np.array(y_test.tolist() + y.tolist())    
             
                #print("-> Data normalization...\n")
                for i in range(X_train.shape[1]):
                    min_max, X_train[:,i] = normalize(X_train[:,i])
                    X_otim[:,i] = normalize(X_otim[:,i], min_max[0], min_max[1])
                    X_test[:,i] = normalize(X_test[:,i], min_max[0], min_max[1])
            
                # -> SVM --------------------------------------------------------------------------
                #print("-> SVM learning and otimization...")
                from sklearn.svm import SVC
                kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
                c_list = frange(1, 10, 0.5)
                otimization_svm = np.ones((len(kernel_list), len(c_list)))
                for _kernel in list(enumerate(kernel_list)):
                    for _c in list(enumerate(c_list)):
                        svm = SVC(C=_c[1], kernel=_kernel[1])
                        scores = cross_val_score(svm, X_otim, y_otim, cv=5)
                        otimization_svm[_kernel[0], _c[0]] = scores.mean()
                
                #print("SVM score = {}".format(np.amax(otimization_svm)))
                result = np.where(otimization_svm == np.amax(otimization_svm))
                listOfCordinates = list(zip(result[0], result[1]))
                best_kernel = kernel_list[listOfCordinates[0][0]]
                best_c = c_list[listOfCordinates[0][1]]
                
                #print("Otimization result -> kernel={} and C={}\n".format(best_kernel, best_c))
            
                svm = SVC(C=best_c, kernel=best_kernel, probability=True)
                svm.fit(X_train, y_train)
                y_pred_svm = svm.predict(X_test)
                probs_svm = svm.predict_proba(X_test)
                probs_svm = probs_svm[:, 1]
                
                a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_svm, probs_svm)
                APCER[0, unseen_test] = a
                BPCER[0, unseen_test] = b 
                BPCER25[0, unseen_test] = c 
                DETC_svm_attack.append(d)
                EER[0, unseen_test] = e 
                AUROC[0, unseen_test] = f
                
                # -> NAIVE BAYES ------------------------------------------------------------------
                #print("-> Naive Bayes learning...")
                from sklearn.naive_bayes import GaussianNB
                nb = GaussianNB()
                scores = cross_val_score(nb, X_otim, y_otim, cv=5)
                #print("NB score = {}\n".format(scores.mean()))
                nb.fit(X_train, y_train)
                y_pred_nb = nb.predict(X_test)
                probs_nb = nb.predict_proba(X_test)
                probs_nb = probs_nb[:, 1]  
                
                a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_nb, probs_nb)
                APCER[1, unseen_test] = a
                BPCER[1, unseen_test] = b 
                BPCER25[1, unseen_test] = c 
                DETC_nb_attack.append(d) 
                EER[1, unseen_test] = e 
                AUROC[1, unseen_test] = f
                
                # -> KNN ------------------------------------------------------------------
                #print("-> KNN learning...")
                from sklearn.neighbors import KNeighborsClassifier
                weights_list = ['uniform', 'distance']
                k_list = range(1, 10)
                otimization_knn = np.ones((len(k_list), len(weights_list)))
                for _w in list(enumerate(weights_list)):
                    for _k in list(enumerate(k_list)):
                        knn = KNeighborsClassifier(n_neighbors=_k[1], weights=_w[1])
                        scores = cross_val_score(knn, X_otim, y_otim, cv=5)
                        otimization_knn[_k[0], _w[0]] = scores.mean()
                
                #print("KNN score = {}".format(np.amax(otimization_knn)))
                result = np.where(otimization_knn == np.amax(otimization_knn))
                listOfCordinates = list(zip(result[0], result[1]))
                best_k = k_list[listOfCordinates[0][0]]
                best_weight = weights_list[listOfCordinates[0][1]]
                
                #print("Otimization result -> k={} and weight={}\n".format(best_k, best_weight))
                #print("------------------------------")
                
                knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
                knn.fit(X_train, y_train)
                y_pred_knn = knn.predict(X_test)
                probs_knn = knn.predict_proba(X_test)
                probs_knn = probs_knn[:, 1]  
                
                a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_knn, probs_knn)
                APCER[2, unseen_test] = a
                BPCER[2, unseen_test] = b
                BPCER25[2, unseen_test] = c
                DETC_knn_attack.append(d)
                EER[2, unseen_test] = e
                AUROC[2, unseen_test] = f 
                
                
        
        APCER_list.append(APCER)
        BPCER_list.append(BPCER)
        BPCER25_list.append(BPCER25)
        EER_list.append(EER)
        AUROC_list.append(AUROC)
        
        DETC_svm_otim.append(DETC_svm_attack)
        DETC_nb_otim.append(DETC_nb_attack)
        DETC_knn_otim.append(DETC_knn_attack)
        
    DETC.append(DETC_svm_otim)
    DETC.append(DETC_nb_otim)
    DETC.append(DETC_knn_otim)
    
        

    APCER = np.nanmean( np.array(APCER_list), axis=0 )
    BPCER = np.nanmean( np.array(BPCER_list), axis=0 )
    BPCER25 = np.nanmean( np.array(BPCER25_list), axis=0 )
    EER = np.nanmean( np.array(EER_list), axis=0 )
    AUROC = np.nanmean( np.array( AUROC_list), axis=0 )
        
    np.savetxt("../Results/" + DATASET + "/Unseen_validation/" + seg  + "UN_VAL_APCER_"+ txt, APCER, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_validation/" + seg  + "UN_VAL_BPCER_"+ txt, BPCER, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_validation/" + seg  + "UN_VAL_BPCER25_"+ txt, BPCER25, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_validation/" + seg  + "UN_VAL_AUROC_"+ txt, AUROC, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_validation/" + seg  + "UN_VAL_EER_"+ txt, EER, fmt="%.3f")

    metrics = [APCER, BPCER, BPCER25, AUROC, EER, DETC]
    
    
    return metrics


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
def getPlots(DETC, num_materials, path):

    
    svm = DETC[0]
    nb = DETC[1]
    knn = DETC[2]
    
    from matplotlib import pyplot as plt
    plt.ioff()

    for k in range(num_materials-1):
    
        fig,ax = plt.subplots(2,num_materials)
        fig.suptitle("{}/AttackMaterial_{}/LBP".format(path, k+1))
        
        for i in range(num_materials):
        
            x1 = svm[i][k][0] #fpr
            y1 = svm[i][k][1] #fnr
            z1 = svm[i][k][2] #tpr
            
            x2 = nb[i][k][0]
            y2 = nb[i][k][1]
            z2 = nb[i][k][2] 
            
            x3 = knn[i][k][0]
            y3 = knn[i][k][1]
            z3 = knn[i][k][2] 
            
            ax[0,i].plot(x1,z1, "-r", label="SVM")
            ax[0,i].plot(x2,z2, "-g", label="NB")
            ax[0,i].plot(x3,z3, "-b", label="KNN")
            ax[0,i].plot([0, 1], [0, 1], color='black', linestyle='--')
            ax[0,i].set_title('ROC - Validation with material {}'.format(i+1))
            ax[0,i].legend(loc="lower right")
            
            ax[1,i].axis_min = min(x1[0],y1[-1])
            ax[1,i].plot(x1,y1, "-r", label="SVM")
            ax[1,i].plot(x2,y2, "-g", label="NB")
            ax[1,i].plot(x3,y3, "-b", label="KNN")
            ax[1,i].set_title('DET - Validation with material {}'.format(i+1))
            ax[1,i].legend(loc="upper right")
            
            
        for a in ax[0,:]:
            a.set(xlabel='', ylabel='True Positive Rate')
        
          
        from matplotlib.ticker import FormatStrFormatter   
        ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2]    
        for a in ax[1,:]:
            a.set(xlabel='BPCER (False Positive Rate)', ylabel='APCER (False Negative Rate)')
            a.label_outer()
            a.set_yscale('log')
            a.set_xscale('log')
            a.get_xaxis().set_major_formatter(FormatStrFormatter('%.2f'))
            a.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
            a.set_xticks(ticks_to_use)
            a.set_yticks(ticks_to_use)
            a.axis([0.001,2,0.001,2])
             
            
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

import datetime

print(datetime.datetime.now()) 

# CrossMatch | Digital_Persona | GreenBit | Hi_Scan | Time_Series
    
###CrossMatch_metrics = unseenValidation('CrossMatch_TESTS', croped=False)
###CrossMatch_metrics = unseenValidation('Digital_Persona_TESTS', croped=False)
 
 
CrossMatch_metrics = unseenValidation('CrossMatch', croped=False)
Digital_Persona_metrics = unseenValidation('Digital_Persona', croped=False)  
GreenBit_metrics = unseenValidation('GreenBit', croped=False)  
Hi_Scan_metrics = unseenValidation('Hi_Scan', croped=False)
Time_Series_metrics = unseenValidation('Time_Series', croped=False)

CrossMatch_seg_metrics = unseenValidation('CrossMatch', croped=True) 
Hi_Scan_seg_metrics = unseenValidation('Hi_Scan', croped=True)
Time_Series_seg_metrics = unseenValidation('Time_Series', croped=True)

####################################################################################################

getPlots(CrossMatch_metrics[5], 3, 'CrossMatch/Unseen_validation/Without_segmentation')
getPlots(Digital_Persona_metrics[5], 4, 'Digital_Persona/Unseen_validation/Without_segmentation')
getPlots(GreenBit_metrics[5], 4, 'GreenBit/Unseen_validation/Without_segmentation')
getPlots(Hi_Scan_metrics[5], 4, 'Hi_Scan/Unseen_validation/Without_segmentation')
getPlots(Time_Series_metrics[5], 3, 'Time_Series/Unseen_validation/Without_segmentation')

getPlots(CrossMatch_seg_metrics[5], 3, 'CrossMatch/Unseen_validation/Segmentation')
getPlots(Hi_Scan_seg_metrics[5], 4, 'Hi_Scan/Unseen_validation/Segmentation')
getPlots(Time_Series_seg_metrics[5], 3, 'Time_Series/Unseen_validation/Segmentation')

print(datetime.datetime.now()) 
