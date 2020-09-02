from PIL import Image
import glob
from skimage import color
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score
import math
    
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

def frange(start, stop=None, step=None):

    num = start
    _list = []
    while num <= stop:
        _list.append(num)
        num = num + step
    
    return _list

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

def oneAttack(DATASET, croped=False):
    
    PATH = 'C:/Users/Asus/Desktop/5_Ano/5_FPAD/Dataset/'
  
    if DATASET=='CrossMatch':
        files = '/*.bmp'
        num_img = 500
        materials = ['Body_Double', 'Ecoflex', 'Playdoh']
    elif DATASET=='Digital_Persona':
        files = '/*.png'
        num_img = 250
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    elif DATASET=='GreenBit':
        files = '/*.png'
        num_img = 250
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    elif DATASET=='Hi_Scan':
        files = '/*.bmp'
        num_img = 250
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    else: #Time_Series
        files = '/*.bmp'
        num_img = 1500
        materials = ['Body_Double', 'Ecoflex', 'Playdoh']
    
    if croped == False:
        print("\nRunning one-attack on {}...\n".format(DATASET))
        live_path = '/Live'
        fake_path = '/Fake/'
        seg = 'Without_segmentation/'
        #txt = DATASET + '_withoutSegmentation_Int.txt'
        #txt = DATASET + '_withoutSegmentation_LBP.txt'
        txt = DATASET + '_withoutSegmentation_IntLBP.txt'
    else:
        files = '/*.png'
        print("\nRunning one-attack with segmentation on {}...\n".format(DATASET))
        live_path = '/Live_c'
        fake_path = '/Fake_c/'
        seg = 'Segmentation/'
        #txt = DATASET + '_withSegmentation_Int.txt'
        #txt = DATASET + '_withSegmentation_LBP.txt'
        txt = DATASET + '_withSegmentation_IntLBP.txt'
       
    
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
    #X = LBP_feature
    X = np.hstack((intensity_feature, LBP_feature))


    y = label

    APCER_list = []
    BPCER_list = []
    BPCER25_list = []
    DETC_svm_list = []
    DETC_nb_list = []
    DETC_knn_list = []
    EER_list = []
    AUROC_list = []
    
    for iteration in range(5):
        
        print("Iteration number {}".format(iteration+1))
    
        APCER = np.ones((num_clf, num_materials))
        BPCER = np.ones((num_clf, num_materials))
        BPCER25 = np.ones((num_clf, num_materials))
        EER = np.ones((num_clf, num_materials))
        AUROC = np.ones((num_clf, num_materials))
        
        DETC_svm = []
        DETC_nb = []
        DETC_knn = []
        
        c = np.array(random.sample(list(zip(X, y)), num_img))
        X, y = zip(*c)
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
        
        for attack in range(len(materials)):
            
            material=materials[attack]
            
            #print("\n*ATTACK NUMBER {} OF {}*\n".format(attack, len(materials)))
            #print("Processing spoof images...\n")
            
            intensity_feature = []
            LBP_feature = []
            label = []
            
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
            
            #S = intensity_feature
            #S = LBP_feature
            S = np.hstack((intensity_feature, LBP_feature))
            
            l = label
               
            S_train, S_test, sl_train, sl_test = train_test_split(S, l, test_size=0.33, random_state=42)
    
         
            #print("------------------------------")
            #print("-> Concluding data preparation...\n")
    
            X_train = np.vstack((X_train, S_train))
            y_train = np.array(y_train.tolist() + sl_train.tolist())
            X_test = np.vstack((X_test, S_test))
            y_test = np.array(y_test.tolist() + sl_test.tolist())   
    
            #print("-> Data normalization...\n")
            for i in range(X_train.shape[1]):
                min_max, X_train[:,i] = normalize(X_train[:,i])
                X_test[:,i] = normalize(X_test[:,i], min_max[0], min_max[1])             
    
                
            # -> SVM --------------------------------------------------------------------------
            #print("-> SVM learning and otimization...")
            
            from sklearn.svm import SVC
            from sklearn.decomposition import PCA
            
            pca = PCA()
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            
            kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
            c_list = list(range(-2 ,2))
            d_list = [1, 2, 3, 4, 5]
            _g = 1/(X_train.shape[1])
            otimization_svm = np.ones((len(kernel_list), len(d_list), len(c_list)))
            for _kernel in list(enumerate(kernel_list)):
                for _d in list(enumerate(d_list)):
                    for _c in list(enumerate(c_list)):
                        #print("Combination: kernel={}, C={}, degree={}".format(_kernel[1], _c[1], _d[1]))
                        svm = SVC(C=math.pow(10, _c[1]), kernel=_kernel[1], degree=_d[1], gamma = 'auto')
                        scores = cross_val_score(svm, X_train, y_train, cv=5)
                        otimization_svm[_kernel[0], _d[0], _c[0]] = scores.mean()
        
        
            result = np.where(otimization_svm == np.amax(otimization_svm))
            listOfCordinates = list(zip(result[0], result[1], result[2]))
            best_kernel = kernel_list[listOfCordinates[0][0]]
            best_d = d_list[listOfCordinates[0][1]]
            best_c = c_list[listOfCordinates[0][2]]
            
            #print("SVM -> kernel={}, C={} and degree={}".format(best_kernel, best_c, best_d))
            
            svm = SVC(C=math.pow(10, best_c), kernel=best_kernel, degree=best_d, gamma = 'auto', probability=True)
            
            svm.fit(X_train, y_train)
            
            y_train_svm = svm.predict(X_train)
            probs_train_svm = svm.predict_proba(X_train)
            probs_train_svm = probs_train_svm[:, 1]
            apcer, bpcer, c, d, eer, f = fpad_iso_metrics(y_train, y_train_svm, probs_train_svm)
            
            y_pred_svm = svm.predict(X_test)
            probs_svm = svm.predict_proba(X_test)
            probs_svm = probs_svm[:, 1]  
            
            #print("Train:\APCER={}\nBPCER={}\nEER={}".format(apcer, bpcer, eer))
            
            '''
            from sklearn.svm import SVC
            from sklearn.model_selection import GridSearchCV
            c_ = list(range(-2 ,2))
            c_list = []
            for c in c_:
                c_list.append(math.pow(10, c))
            d_list = [1, 2, 3, 4, 5]
            parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':c_list, 'degree':d_list}
            svc = SVC(probability=True)
            svm = GridSearchCV(svc, parameters)
            svm.fit(X_train, y_train)
            y_pred_svm = svm.predict(X_test)
            probs_svm = svm.predict_proba(X_test)
            probs_svm = probs_svm[:, 1]  
            '''
            a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_svm, probs_svm)
            APCER[0, attack] = a
            BPCER[0, attack] = b 
            BPCER25[0, attack] = c 
            DETC_svm.append(d)
            EER[0, attack] = e 
            AUROC[0, attack] = f
            '''            
            # -> NAIVE BAYES ------------------------------------------------------------------
            #print("-> Naive Bayes learning...")
            from sklearn.naive_bayes import GaussianNB
            nb = GaussianNB()
            scores = cross_val_score(nb, X_train, y_train, cv=5)
            #print("NB score = {}\n".format(scores.mean()))
            nb.fit(X_train, y_train)
            y_pred_nb = nb.predict(X_test)
            probs_nb = nb.predict_proba(X_test)
            probs_nb = probs_nb[:, 1]  
            
            a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_nb, probs_nb)
            APCER[1, attack] = a
            BPCER[1, attack] = b 
            BPCER25[1, attack] = c 
            DETC_nb.append(d) 
            EER[1, attack] = e 
            AUROC[1, attack] = f
            
            # -> KNN ------------------------------------------------------------------
            #print("-> KNN learning...")
            from sklearn.neighbors import KNeighborsClassifier
            weights_list = ['uniform', 'distance']
            k_list = range(1, 10)
            otimization_knn = np.ones((len(k_list), len(weights_list)))
            for _w in list(enumerate(weights_list)):
                for _k in list(enumerate(k_list)):
                    knn = KNeighborsClassifier(n_neighbors=_k[1], weights=_w[1])
                    scores = cross_val_score(knn, X_train, y_train, cv=5)
                    otimization_knn[_k[0], _w[0]] = scores.mean()
            
            #print("KNN score = {}".format(np.amax(otimization_knn)))
            result = np.where(otimization_knn == np.amax(otimization_knn))
            listOfCordinates = list(zip(result[0], result[1]))
            best_k = k_list[listOfCordinates[0][0]]
            best_weight = weights_list[listOfCordinates[0][1]]
            
            #print("KNN -> k={} and weight={}".format(best_k, best_weight))
            #print("------------------------------")
            
            knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
            knn.fit(X_train, y_train)
            y_pred_knn = knn.predict(X_test)
            probs_knn = knn.predict_proba(X_test)
            probs_knn = probs_knn[:, 1]  
            
            a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_knn, probs_knn)
            APCER[2, attack] = a
            BPCER[2, attack] = b
            BPCER25[2, attack] = c
            DETC_knn.append(d)
            EER[2, attack] = e
            AUROC[2, attack] = f
            '''            

        APCER_list.append(APCER)
        BPCER_list.append(BPCER)
        BPCER25_list.append(BPCER25)
        DETC_svm_list.append(DETC_svm)
        DETC_nb_list.append(DETC_nb)
        DETC_knn_list.append(DETC_knn)
        EER_list.append(EER)
        AUROC_list.append(AUROC)
        
        
    DETC = []
    DETC.append(DETC_svm_list)
    DETC.append(DETC_nb_list)
    DETC.append(DETC_knn_list)

    
    APCER = np.nanmean( np.array([ APCER_list[0], APCER_list[1], APCER_list[2], APCER_list[3], APCER_list[4] ]), axis=0 )
    BPCER = np.nanmean( np.array([ BPCER_list[0], BPCER_list[1], BPCER_list[2], BPCER_list[3], BPCER_list[4] ]), axis=0 )
    BPCER25 = np.nanmean( np.array([ BPCER25_list[0], BPCER25_list[1], BPCER25_list[2], BPCER25_list[3], BPCER25_list[4] ]), axis=0 )
    EER = np.nanmean( np.array([ EER_list[0], EER_list[1], EER_list[2], EER_list[3], EER_list[4] ]), axis=0 )
    AUROC = np.nanmean( np.array([ AUROC_list[0], AUROC_list[1], AUROC_list[2], AUROC_list[3], AUROC_list[4] ]), axis=0 )
    
    '''
    np.savetxt("../Results/" + DATASET + "/One_attack/" + seg  + "ONE_APCER_"+ txt, APCER, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/One_attack/" + seg  + "ONE_BPCER_"+ txt, BPCER, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/One_attack/" + seg  + "ONE_BPCER25_"+ txt, BPCER25, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/One_attack/" + seg  + "ONE_AUROC_"+ txt, AUROC, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/One_attack/" + seg  + "ONE_EER_"+ txt, EER, fmt="%.3f")
    '''
    metrics = [APCER, BPCER, BPCER25, AUROC, EER, DETC]
    
    return metrics


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
def getPlots(DETC, num_materials, path):

    
    svm = DETC[0]
    nb = DETC[1]
    knn = DETC[2]
    
    from matplotlib import pyplot as plt

    fig,ax = plt.subplots(2,num_materials)
    fig.suptitle("{}/Int".format(path))
    
    for i in range(num_materials):
    
        x1 = svm[0][i][0] #fpr
        y1 = svm[0][i][1] #fnr
        z1 = svm[0][i][2] #tpr
        
        x2 = nb[0][i][0]
        y2 = nb[0][i][1]
        z2 = nb[0][i][2] 
        
        x3 = knn[0][i][0]
        y3 = knn[0][i][1]
        z3 = knn[0][i][2] 
        
        ax[0,i].plot(x1,z1, "-r", label="SVM")
        ax[0,i].plot(x2,z2, "-g", label="NB")
        ax[0,i].plot(x3,z3, "-b", label="KNN")
        ax[0,i].plot([0, 1], [0, 1], color='black', linestyle='--')
        ax[0,i].set_title('ROC - Material {}'.format(i+1))
        ax[0,i].legend(loc="lower right")
        
        ax[1,i].axis_min = min(x1[0],y1[-1])
        ax[1,i].plot(x1,y1, "-r", label="SVM")
        ax[1,i].plot(x2,y2, "-g", label="NB")
        ax[1,i].plot(x3,y3, "-b", label="KNN")
        ax[1,i].set_title('DET - Material {}'.format(i+1))
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

# CrossMatch | Digital_Persona | GreenBit | Hi_Scan | Time_Series
    
###CrossMatch_metrics = oneAttack('CrossMatch_TESTS', croped=False)
###CrossMatch_metrics = oneAttack('Digital_Persona_TESTS', croped=False)
    
import datetime

print(datetime.datetime.now())

CrossMatch_metrics = oneAttack('CrossMatch', croped=False)
#Digital_Persona_metrics = oneAttack('Digital_Persona', croped=False)  
#GreenBit_metrics = oneAttack('GreenBit', croped=False)  
#Hi_Scan_metrics = oneAttack('Hi_Scan', croped=False)
#Time_Series_metrics = oneAttack('Time_Series', croped=False)

#CrossMatch_seg_metrics = oneAttack('CrossMatch', croped=True) 
#Hi_Scan_seg_metrics = oneAttack('Hi_Scan', croped=True)
#Time_Series_seg_metrics = oneAttack('Time_Series', croped=True)
   
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#'DATASET/One_attack/Without_segmentation'
#'DATASET/One_attack/Segmentation'
'''
getPlots(CrossMatch_metrics[5], 3, 'CrossMatch/One_attack/Without_segmentation')
getPlots(Digital_Persona_metrics[5], 4, 'Digital_Persona/One_attack/Without_segmentation')
getPlots(GreenBit_metrics[5], 4, 'GreenBit/One_attack/Without_segmentation')
getPlots(Hi_Scan_metrics[5], 4, 'Hi_Scan/One_attack/Without_segmentation')
getPlots(Time_Series_metrics[5], 3, 'Time_Series/One_attack/Without_segmentation')

getPlots(CrossMatch_seg_metrics[5], 3, 'CrossMatch/One_attack/Segmentation')
getPlots(Hi_Scan_seg_metrics[5], 4, 'Hi_Scan/One_attack/Segmentation')
getPlots(Time_Series_seg_metrics[5], 3, 'Time_Series/One_attack/Segmentation')
'''
print(datetime.datetime.now())
