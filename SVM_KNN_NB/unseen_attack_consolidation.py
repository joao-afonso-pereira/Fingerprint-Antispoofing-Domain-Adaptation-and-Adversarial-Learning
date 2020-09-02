from __future__ import division

from PIL import Image
import glob
from skimage import color
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score
import math
from sklearn.metrics import accuracy_score

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

from scipy.signal import convolve2d

def lpq(img, winSize=3, freqestim=1, mode='nh'):
    rho=0.90

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
        LPQdesc=LPQdesc/LPQdesc.sum()

    return LPQdesc
    
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
    
    BPCER1=fpr[(np.abs(fnr - 0.01)).argmin()] #closest to 4
    BPCER5=fpr[(np.abs(fnr - 0.05)).argmin()] #closest to 4
    BPCER10=fpr[(np.abs(fnr - 0.1)).argmin()] #closest to 4

    APCER1=fnr[(np.abs(fnr - 0.01)).argmin()] #closest to 4
    APCER5=fnr[(np.abs(fnr - 0.05)).argmin()] #closest to 4
    APCER10=fnr[(np.abs(fnr - 0.1)).argmin()] #closest to 4
            
    DETC = [fpr, fnr, tpr]    
    
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    AUROC = roc_auc_score(ground_truth, probs)
    
    Acc = accuracy_score(ground_truth, prediction)
      
    return(Acc, APCER, BPCER, EER, BPCER1, BPCER5, BPCER10, APCER1, APCER5, APCER10)

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

def unseenAttack(DATASET, croped=False):
    
    PATH = 'L:/FPAD/Dataset/'
  
    if DATASET=='CrossMatch':
        files = '/*.bmp'
        percentage = 0.33
        materials = ['Body_Double', 'Ecoflex', 'Playdoh']
    elif DATASET=='Digital_Persona':
        files = '/*.png'
        percentage = 0.25
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    elif DATASET=='GreenBit':
        files = '/*.png'
        percentage = 0.25
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    elif DATASET=='Hi_Scan':
        files = '/*.bmp'
        percentage = 0.25
        materials = ['Ecoflex_00_50', 'Gelatine', 'Latex', 'WoodGlue']
    else: #Time_Series
        files = '/*.bmp'
        percentage = 0.33
        materials = ['Body_Double', 'Ecoflex', 'Playdoh']
    
    if croped == False:
        print("\nRunning unseen-attack on {}...\n".format(DATASET))
        live_path = '/Live'
        fake_path = '/Fake/'
        seg = 'Without_segmentation/'
        #txt = DATASET + '_withoutSegmentation_Int.txt'
        #txt = DATASET + '_withoutSegmentation_LBP.txt'
        txt = DATASET + '_withoutSegmentation_IntLBP.txt'
    else:
        files = '/*.png'
        print("\nRunning unseen-attack with segmentation on {}...\n".format(DATASET))
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
    LPQ_feature = []
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
            LPQ_feature.append(lpq(img))
            label.append(0)
        

    intensity_feature = np.array(intensity_feature)
    LBP_feature = np.array(LBP_feature)
    LPQ_feature = np.array(LPQ_feature)
    label = np.array(label)
      
    #X = intensity_feature
    #X = LBP_feature
    X = np.hstack((intensity_feature, LBP_feature, LPQ_feature))

    y = label
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentage, random_state=42)
    
    Accuracy = np.ones((num_clf, num_materials))
    APCER = np.ones((num_clf, num_materials))
    BPCER = np.ones((num_clf, num_materials))
    EER = np.ones((num_clf, num_materials))
    AUROC = np.ones((num_clf, num_materials))
    BPCER1 = np.ones((num_clf, num_materials))
    BPCER5 = np.ones((num_clf, num_materials))
    BPCER10 = np.ones((num_clf, num_materials))
    APCER1 = np.ones((num_clf, num_materials))
    APCER5 = np.ones((num_clf, num_materials))
    APCER10 = np.ones((num_clf, num_materials))
    
    DETC_svm = []
    DETC_nb = []
    DETC_knn = []

    
    # TEST/TRAIN SPOOF + CLASSIFICATION + METRICS ------------------------------------------------------------
    
    for unseen in range(0, len(materials)):
        
        print("\n*UNSEEN ATTACK NUMBER {} OF {}*\n".format(unseen+1, len(materials)))
        #print("Processing spoof images...\n")
        
        intensity_attack = []
        LBP_attack = []
        label_attack = []
        LPQ_attack = []
        
        for k in range(0, len(materials)): 
           
            if k==unseen:
                    
                material=materials[k]
            
                #print("\n*ATTACK NUMBER {} OF {}*\n".format(attack, len(materials)))
                #print("Processing spoof images...\n")
                
                intensity_test = []
                LBP_test = []
                LPQ_test = []
                label_test = []
                
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
                        LPQ_test.append(lpq(img))
                        label_test.append(1)
                
                intensity_test = np.array(intensity_test)
                LBP_test = np.array(LBP_test)
                LPQ_test = np.array(LPQ_test)
                label_test = np.array(label_test)     
                
            else:
                
                material=materials[k]
            
                #print("\n*ATTACK NUMBER {} OF {}*\n".format(attack, len(materials)))
                #print("Processing spoof images...\n")
                
                for filename in glob.glob(PATH + 'LivDet2015/train/' + DATASET + fake_path + material + files):
                    with Image.open(filename) as img:
                        if DATASET=='Digital_Persona':
                            img = color.rgb2gray(np.array(img))
                        else:
                            img = np.array(img)
                        hist1, bin_edges1 = np.histogram(img, density=True)
                        intensity_attack.append(hist1.tolist())
                        lbp = local_binary_pattern(img, n_points, radius, METHOD)
                        hist2, bin_edges2 = np.histogram(lbp, density=True)
                        LBP_attack.append(hist2.tolist())
                        LPQ_attack.append(lpq(img))
                        label_attack.append(1)
                
                    
        #print("------------------------------")
        #print("-> Concluding data preparation...\n")
        #################################################
        
        intensity_attack = np.array(intensity_attack)
        LBP_attack = np.array(LBP_attack)
        LPQ_attack = np.array(LPQ_attack)
        label_attack = np.array(label_attack)   
        
        #X = intensity_attack
        #X = LBP_attack
        X = np.hstack((intensity_attack, LBP_attack, LPQ_attack))
        
        X_train = np.vstack((X_train, X))
        y = label_attack
        y_train = np.array(y_train.tolist() + y.tolist())
        
        #################################################
        intensity_test = np.array(intensity_test)
        LBP_test = np.array(LBP_test)
        LPQ_test = np.array(LPQ_test)
        label_test = np.array(label_test)   
        
        #X = intensity_test
        #X = LBP_test
        X = np.hstack((intensity_test, LBP_test, LPQ_test))
        
        X_test = np.vstack((X_test, X))
        
        y = label_test
        y_test = np.array(y_test.tolist() + y.tolist())    
        
        #print("-> Data normalization...\n")
        for i in range(X_train.shape[1]):
            min_max, X_train[:,i] = normalize(X_train[:,i])
            X_test[:,i] = normalize(X_test[:,i], min_max[0], min_max[1]) 
        
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
        svm.score(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        probs_svm = svm.predict_proba(X_test)
        probs_svm = probs_svm[:, 1]  
    
        a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_svm, probs_svm)
        APCER[0, unseen] = a
        BPCER[0, unseen] = b 
        BPCER25[0, unseen] = c 
        DETC_svm.append(d)
        EER[0, unseen] = e 
        AUROC[0, unseen] = f
        '''
        
        
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
        
        print("SVM -> kernel={}, C={} and degree={}".format(best_kernel, best_c, best_d))
        
        svm = SVC(C=math.pow(10, best_c), kernel=best_kernel, degree=best_d, gamma = 'auto', probability=True)
        
        svm.fit(X_train, y_train)
        
        y_train_svm = svm.predict(X_train)
        probs_train_svm = svm.predict_proba(X_train)
        probs_train_svm = probs_train_svm[:, 1]
        a, b, c, d, e, f, g, h, i, j = fpad_iso_metrics(y_train, y_train_svm, probs_train_svm)
        
        y_pred_svm = svm.predict(X_test)
        probs_svm = svm.predict_proba(X_test)
        probs_svm = probs_svm[:, 1]  
        
        print("Train Accuracy = {}".format(a))
        
        
        y_pred_svm = svm.predict(X_test)
        probs_svm = svm.predict_proba(X_test)
        probs_svm = probs_svm[:, 1]
        
        a, b, c, d, e, f, g, h, i, j = fpad_iso_metrics(y_test, y_pred_svm, probs_svm)
        Accuracy[0, unseen] = a
        APCER[0, unseen] = b
        BPCER[0, unseen] = c
        EER[0, unseen] = d 
        BPCER1[0, unseen] = e
        BPCER5[0, unseen] = f
        BPCER10[0, unseen] = g
        APCER1[0, unseen] = h
        APCER5[0, unseen] = i
        APCER10[0, unseen] = j
        
        
        '''
        # -> NAIVE BAYES ------------------------------------------------------------------
        #print("-> Naive Bayes learning...")
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
        scores = cross_val_score(nb, X_train, y_train, cv=5)
        #print("NB score = {}".format(scores.mean()))
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        probs_nb = nb.predict_proba(X_test)
        probs_nb = probs_nb[:, 1]  
        
        a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_nb, probs_nb)
        APCER[1, unseen] = a
        BPCER[1, unseen] = b 
        BPCER25[1, unseen] = c 
        DETC_nb.append(d) 
        EER[1, unseen] = e 
        AUROC[1, unseen] = f
        
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
        
        #print("Otimization result -> k={} and weight={}".format(best_k, best_weight))
        #print("------------------------------")
        
        knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        probs_knn = knn.predict_proba(X_test)
        probs_knn = probs_knn[:, 1]  
        
        a, b, c, d, e, f = fpad_iso_metrics(y_test, y_pred_knn, probs_knn)
        APCER[2, unseen] = a
        BPCER[2, unseen] = b
        BPCER25[2, unseen] = c
        DETC_knn.append(d)
        EER[2, unseen] = e
        AUROC[2, unseen] = f
        '''

    DETC = []
    DETC.append(DETC_svm)
    DETC.append(DETC_nb)
    DETC.append(DETC_knn)    
    
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_Acc_"+ txt, Accuracy, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_APCER_"+ txt, APCER, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_BPCER_"+ txt, BPCER, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_EER_"+ txt, EER, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_BPCER1_"+ txt, BPCER1, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_BPCER5_"+ txt, BPCER5, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_BPCER10_"+ txt, BPCER10, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_APCER1_"+ txt, APCER1, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_APCER5_"+ txt, APCER5, fmt="%.3f")
    np.savetxt("../Results/" + DATASET + "/Unseen_attack/" + seg  + "UNSEEN_APCER10_"+ txt, APCER10, fmt="%.3f")
    
    metrics = [Accuracy, APCER, BPCER, EER, BPCER1, BPCER5, BPCER10, APCER1, APCER5, APCER10]
    
    return metrics


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
def getPlots(DETC, num_materials, path):

    
    svm = DETC[0]
    nb = DETC[1]
    knn = DETC[2]
    
    from matplotlib import pyplot as plt
    plt.ioff()
    
    fig,ax = plt.subplots(2,num_materials)
    fig.suptitle("{}/Int".format(path))
    
    for i in range(num_materials):
    
        x1 = svm[i][0] #fpr
        y1 = svm[i][1] #fnr
        z1 = svm[i][2] #tpr
        
        x2 = nb[i][0]
        y2 = nb[i][1]
        z2 = nb[i][2] 
        
        x3 = knn[i][0]
        y3 = knn[i][1]
        z3 = knn[i][2] 
        
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
    
import datetime

# CrossMatch | Digital_Persona | GreenBit | Hi_Scan | Time_Series
    
#CrossMatch_metrics = unseenAttack('CrossMatch_TESTS', croped=False)
#CrossMatch_metrics = unseenAttack('Digital_Persona_TESTS', croped=False)

print(datetime.datetime.now()) 

#CrossMatch_metrics_UNSEEN = unseenAttack('CrossMatch', croped=False)
#Digital_Persona_metrics = unseenAttack('Digital_Persona', croped=False)  
#GreenBit_metrics = unseenAttack('GreenBit', croped=False)  
#Hi_Scan_metrics = unseenAttack('Hi_Scan', croped=False)
Time_Series_metrics = unseenAttack('Time_Series', croped=False)


CrossMatch_seg_metrics = unseenAttack('CrossMatch', croped=True) 
Digital_Persona_metrics = unseenAttack('Digital_Persona', croped=True)  
GreenBit_metrics = unseenAttack('GreenBit', croped=True) 
Hi_Scan_seg_metrics = unseenAttack('Hi_Scan', croped=True)
Time_Series_seg_metrics = unseenAttack('Time_Series', croped=True)


'''
getPlots(CrossMatch_metrics[5], 3, 'CrossMatch/Unseen_attack/Without_segmentation')
getPlots(Digital_Persona_metrics[5], 4, 'Digital_Persona/Unseen_attack/Without_segmentation')
getPlots(GreenBit_metrics[5], 4, 'GreenBit/Unseen_attack/Without_segmentation')
getPlots(Hi_Scan_metrics[5], 4, 'Hi_Scan/Unseen_attack/Without_segmentation')
getPlots(Time_Series_metrics[5], 3, 'Time_Series/Unseen_attack/Without_segmentation')

getPlots(CrossMatch_seg_metrics[5], 3, 'CrossMatch/Unseen_attack/Segmentation')
getPlots(Hi_Scan_seg_metrics[5], 4, 'Hi_Scan/Unseen_attack/Segmentation')
getPlots(Time_Series_seg_metrics[5], 3, 'Time_Series/Unseen_attack/Segmentation')
'''
print(datetime.datetime.now())