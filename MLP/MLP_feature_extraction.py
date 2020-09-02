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
import sys

from scipy.signal import convolve2d

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


def featureExtractor(DATASET, croped=False):

    PATH = "L:/FPAD/Dataset/LivDet2015/train/"
    
    radius = 1
    n_points = 8 * radius
    METHOD = 'uniform' 
       
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
    elif DATASET=='Time_Series': #Time_Series
        files = '/*.bmp'
        num_img = 1500
        materials = ['Body_Double', 'Ecoflex', 'Playdoh']
    else:
        sys.exit("Error: incorrect dataset!")
    
    if croped == False:
        print("\nExtracting features on {}...".format(DATASET))
        live_path = '/Live'
        fake_path = '/Fake/'
        seg = 'Without_segmentation/'
        #txt = DATASET + '_withoutSegmentation_Int.txt'
        #txt = DATASET + '_withoutSegmentation_LBP.txt'
        txt = DATASET + '_withoutSegmentation_IntLBP.txt'
        csv = ""
    else:
        files = '/*.png'
        print("\nExtracting features of ROI on {}...".format(DATASET))
        live_path = '/Live_c'
        fake_path = '/Fake_c/'
        seg = 'Segmentation/'
        #txt = DATASET + '_withSegmentation_Int.txt'
        #txt = DATASET + '_withSegmentation_LBP.txt'
        txt = DATASET + '_withSegmentation_IntLBP.txt'
        csv = "_c"
       
    
    num_clf = 3 
    num_materials = (len(materials))
    
    F = []
    
    intensity_feature = []
    LBP_feature = []
    LPQ_feature = []
    
    print("\n   - Bonafide samples")
    for filename in glob.glob(PATH + DATASET + live_path + files):
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
    
    
    intensity_feature = np.array(intensity_feature)
    LBP_feature = np.array(LBP_feature)
    LPQ_feature = np.array(LPQ_feature)
    
    BF = np.hstack((intensity_feature, LBP_feature, LPQ_feature))
    
    for i in range(BF.shape[1]):
        min_max, BF[:,i] = normalize(BF[:,i])
        
    F.append(BF)
    
    
    print("\n   - Presentation attack samples")
    for material in materials:
        print("\n      - " + material)
        intensity_feature = []
        LBP_feature = []
        LPQ_feature = []
        for filename in glob.glob(PATH + DATASET + fake_path + material + files):
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
                
        intensity_feature = np.array(intensity_feature)
        LBP_feature = np.array(LBP_feature)
        LPQ_feature = np.array(LPQ_feature)  
        
        PAI = np.hstack((intensity_feature, LBP_feature, LPQ_feature)) 
        
        for i in range(PAI.shape[1]):
            PAI[:,i] = normalize(PAI[:,i], min_max[0], min_max[1])
            
        F.append(PAI)
        
    import pandas as pd 
    pd.DataFrame(F[0]).to_csv(DATASET + '_real' + csv + '.csv')

    for i in range(1, len(F)):
        pd.DataFrame(F[i]).to_csv(DATASET + '_' + materials[i-1] + csv + '.csv')


'''
datasets = ['CrossMatch', 'Digital_Persona', 'GreenBit', 'Hi_Scan', 'Time_Series']

for dataset in datasets:
    featureExtractor(dataset, croped=False)

'''    
datasets = ['Digital_Persona', 'GreenBit']

for dataset in datasets:
    featureExtractor(dataset, croped=True)
