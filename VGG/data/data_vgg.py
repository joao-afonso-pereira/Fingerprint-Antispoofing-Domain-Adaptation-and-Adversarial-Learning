import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from torchvision import transforms
import sys
import math
from PIL import Image
from PIL import ImageOps
import cv2
import matplotlib.pyplot as plt

class numpyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()
          
BATCH_SIZE = 8
MIN_WIDTH = -1
MIN_HEIGHT = -1

MIN_CROP_WIDTH = -1
MIN_CROP_HEIGHT = -1

def get_data_loaders(path, dataset, test_material, croped=True, unseen_attack=False):
    
    global MIN_WIDTH 
    global MIN_HEIGHT 
    global MIN_CROP_WIDTH 
    global MIN_CROP_HEIGHT 
    global BATCH_SIZE
    
    if unseen_attack==True:
        
        if dataset == "CrossMatch":
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]
            real_train = np.array(range(1000))
            real_test = np.array(range(1000, 1500))
            MIN_WIDTH = 141
            MIN_HEIGHT = 205
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 247
        elif dataset == "Digital_Persona":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.array(range(750))
            real_test = np.array(range(750, 1000))
            MIN_WIDTH = 109
            MIN_HEIGHT = 157
            MIN_CROP_WIDTH = 224
            MIN_CROP_HEIGHT = 235
        elif dataset == "GreenBit":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.array(range(750))
            real_test = np.array(range(750, 997))
            MIN_WIDTH = 101
            MIN_HEIGHT = 149
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 271
        elif dataset == "Hi_Scan":
            BATCH_SIZE = 16
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.array(range(750))
            real_test = np.array(range(750, 1000))
            MIN_WIDTH = 163
            MIN_HEIGHT = 251
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 311
        elif dataset == "Time_Series":
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]
            real_train = np.array(range(2960))
            real_test = np.array(range(2960, 4440))
            MIN_WIDTH = 55
            MIN_HEIGHT = 49
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 225
        else:
            sys.exit("Error: incorrect dataset!")
  
                   
        train_materials = np.delete(materials_list, test_material)
        
        data = FPAD(path, dataset, material_idx=train_materials, real_idx=real_train, croped=croped)
        data_test = FPAD(path, dataset, material_idx=[test_material], real_idx=real_test, croped=croped)
        
        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size
        data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])
   
    else:
        
        if dataset == "CrossMatch":
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]
            real_train = np.random.randint(low=0, high=1500, size=500)
            MIN_WIDTH = 141
            MIN_HEIGHT = 205
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 247
        elif dataset == "Digital_Persona":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.random.randint(low=0, high=1000, size=250)
            MIN_WIDTH = 109
            MIN_HEIGHT = 157
            MIN_CROP_WIDTH = 224
            MIN_CROP_HEIGHT = 235
        elif dataset == "GreenBit":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.random.randint(low=0, high=997, size=250)
            MIN_WIDTH = 101
            MIN_HEIGHT = 149
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 271
        elif dataset == "Hi_Scan":
            BATCH_SIZE = 16
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.random.randint(low=0, high=1000, size=250)
            MIN_WIDTH = 163
            MIN_HEIGHT = 251
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 311
        elif dataset == "Time_Series":
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]
            real_train = np.random.randint(low=0, high=4440, size=1480)
            MIN_WIDTH = 55
            MIN_HEIGHT = 49
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 225
        else:
            sys.exit("Error: incorrect dataset!")        
        
        
        train_materials = [test_material]
        
        dataset = FPAD(path, dataset, material_idx=train_materials, real_idx=real_train, croped=croped)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _dataset, data_test = torch.utils.data.random_split(dataset, [train_size, test_size])
    
        train_size = int(0.8 * len(_dataset))
        val_size = len(_dataset) - train_size
        data_train, data_val = torch.utils.data.random_split(_dataset, [train_size, val_size])
      
    
    print('\n--------------------------------------')
    print('Train materials: ', end="")
    for material in train_materials:
        print(materials_name[material], end="  ")
    print('\nTest material: {}'.format(materials_name[test_material]))
    
    #Data loaders
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0} 
  
    train_loader = torch.utils.data.DataLoader(data_train, **params)
    valid_loader = torch.utils.data.DataLoader(data_val, **params)
    test_loader = torch.utils.data.DataLoader(data_test, **params)

    print('\nDatasets size: Train {}, Val {}, Test {}'.format(len(data_train),
                                                           len(data_val),
                                                           len(data_test)))
  
    return train_loader, valid_loader, test_loader


  
IMG_HEIGHT = -1 
IMG_WIDTH = -1 


class FPAD(Dataset):
    def __init__(self,
                 PATH,
                 dataset,
                 material_idx,
                 real_idx,
                 croped = True):

        self.material_idx = material_idx
        self.real_idx = real_idx
        self.dataset = dataset
        self.croped = croped
        
        if dataset == "CrossMatch" or dataset=="Time_Series":
            self.materials = ["Body_Double", "Ecoflex", "Playdoh"]
        else:
            self.materials = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

        # Initialize X (data), y (real=1, fake=0), and f (fake material id) arrays
        X = []
        y = []
        f = []
        f_norm = []
        
        #PRESENTATION ATTACK SAMPLES
        
        count = 0
    
        for index in self.material_idx:                 
            
            if croped == False:
                self.fake_dir = PATH + dataset + "/Fake/" + self.materials[index] + "/"
                # read fake names
                txt_path = PATH + dataset + "/" + self.materials[index] + ".txt"
            else:
                self.fake_dir = PATH + dataset + "/Fake_c/" + self.materials[index] + "/"
                # read fake names
                txt_path= PATH + dataset + "/" + self.materials[index] + "_c.txt"
            
            with open(txt_path, 'r') as file:
              fake_names = file.readlines()
    
            count = count + len(fake_names)
    
            X.extend(fake_names)
            y.extend([1]*len(fake_names))
            
            f.extend([index]*len(fake_names))
            f_norm.extend([index]*len(fake_names))
    
        self.n_presentation_attack_samples = count

        #BONAFIDE SAMPLES
        
        if croped == False:
            path = PATH + dataset + "/real.txt"
        else:
            path = PATH + dataset + "/real_c.txt"
    
        # read real names
        with open(path, 'r') as file:
          real_names = file.readlines()
          
        real_names = np.array(real_names)
        
        self.n_bonafide_samples = self.real_idx.shape[0]
        
        real_names = real_names[self.real_idx]        

        # append real_data to X, y, and f arrays
        X.extend(real_names)
        y.extend([0]*self.n_bonafide_samples)
        f.extend([-1]*self.n_bonafide_samples)
        f_norm.extend([-1]*self.n_bonafide_samples)

        self.X = np.array(X)
        self.y = np.array(y)
        self.f = np.array(f)
        self.f_norm = np.array(f_norm)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img_name = self.X[idx]

        
        sample = Image.open(img_name.rstrip())
        width, height = sample.size
        
        sample = np.array(sample)/255.0
                      
        if self.croped == True:    
            
            
            sample = Image.fromarray(np.uint8(sample))
            width, height = sample.size
        
            
            left = int((width-MIN_CROP_WIDTH)/2)
            right = width - MIN_CROP_WIDTH - left
            top = int((height-MIN_CROP_HEIGHT)/2)
            bottom = height - MIN_CROP_HEIGHT - top
            
            sample = ImageOps.crop(sample, (left, top, right, bottom))            
            
        transformation = self.transformations()
        
        width, height = sample.size
        
        #Resize
        if width < height:
            ratio = 224/width
        else:
            ratio = 224/height
            
        ratio = math.ceil(ratio)
        
        #sample = sample.resize((width*ratio, height*ratio)) #AO FAZER ISTO NOS CROP SUBSTITUIR NEW_WIDTH POR MIN_WIDTH E NEW_HEIGHT POR MIN_HEIGHT
        
        sample = np.array(sample)
        
        if self.dataset == "Digital_Persona":
            
            sample = sample[:,:,0]
            sample =  np.transpose(sample)
              
    
        sample.reshape((1, sample.shape[0], sample.shape[1]))
        
        sample = cv2.merge((sample, sample, sample))
       
        return transformation(sample).view((3, sample.shape[0], sample.shape[1])), self.y[idx], self.f[idx], self.f_norm[idx]
    
    def transformations(self):
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return data_transform
    
#%%
    
if __name__ == '__main__':

    
    train, val, test = get_data_loaders("L:/FPAD/Dataset/LivDet2015/train/", "GreenBit", 0, croped=True, unseen_attack=True)

    for i, (x, y, f, _) in enumerate(train):
        print(x.shape)
        print(y.shape)
        print(f.shape)
        
        x = x[f==-1]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.axis("off")
        ax1.imshow(x[0,0,...], 'gray')
        ax2.axis("off")
        ax2.imshow(x[1,0,...], 'gray')
        ax3.axis("off")
        ax3.imshow(x[2,0,...], 'gray')
        ax4.axis("off")
        ax4.imshow(x[3,0,...], 'gray')
        fig.show()
        break