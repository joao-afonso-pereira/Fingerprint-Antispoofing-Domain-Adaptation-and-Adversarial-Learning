import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from torchvision import transforms
import sys
from PIL import Image
from PIL import ImageOps


class numpyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()    

MIN_WIDTH = -1
MIN_HEIGHT = -1

MIN_CROP_WIDTH = -1
MIN_CROP_HEIGHT = -1

def get_data_loaders(path, dataset, test_material, img_size, batch_size, croped=True, unseen_attack=True):
    
    global MIN_WIDTH 
    global MIN_HEIGHT 
    global MIN_CROP_WIDTH 
    global MIN_CROP_HEIGHT 
    global BATCH_SIZE

    sensor = dataset
    
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
        
        data = FPAD(path, dataset, img_size, material_idx=train_materials, real_idx=real_train)
        data_test = FPAD(path, dataset, img_size, material_idx=[test_material], real_idx=real_test)
        
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
        
        dataset = FPAD(path, dataset, img_size, material_idx=train_materials, real_idx=real_train)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _dataset, data_test = torch.utils.data.random_split(dataset, [train_size, test_size])
    
        train_size = int(0.8 * len(_dataset))
        val_size = len(_dataset) - train_size
        data_train, data_val = torch.utils.data.random_split(_dataset, [train_size, val_size])
      
    
    print('\n--------------------------------------')
    print('Dataset: ' + sensor)
    print('Train materials: ', end="")
    for material in train_materials:
        print(materials_name[material], end="  ")
    print('\nTest material: {}'.format(materials_name[test_material]))
    
    #Data loaders
    
    params = {'batch_size': batch_size,
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
                 img_size,
                 material_idx,
                 real_idx,
                 croped = True):

        self.material_idx = material_idx
        self.real_idx = real_idx
        self.dataset = dataset
        self.croped = croped
        self.img_size = img_size
        
        if dataset == "CrossMatch" or dataset=="Time_Series":
            self.materials = ["Body_Double", "Ecoflex", "Playdoh"]
        else:
            self.materials = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

        # Initialize X (data), y (real=1, fake=0)
        X = []
        y = []
        
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
            y.extend([float(1)]*len(fake_names))

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

        X.extend(real_names)
        y.extend([float(0)]*self.n_bonafide_samples)

        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img_name = self.X[idx]       
        sample = Image.open(img_name.rstrip())
        width, height = sample.size  

        dim = self.img_size

        if width > height:
            ratio = dim/width
        else:
            ratio = dim/height

        new_width = round(width*ratio)
        new_height = round(height*ratio)

        sample = sample.resize((new_width, new_height))
        width, height = sample.size
        
        delta_w = dim - width
        delta_h = dim - height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        sample = ImageOps.expand(sample, padding, 255)
                 
        #sample = 255 - np.array(sample)
        sample = np.array(sample)
        
        if self.dataset == "Digital_Persona":
            
            sample = sample[:,:,0]
            sample =  np.transpose(sample)
                 
        sample.reshape((1, sample.shape[0], sample.shape[1]))

        transformation = self.transformations()
       
        return transformation(sample).view((1, sample.shape[0], sample.shape[1])), self.y[idx]
    
    def transformations(self):
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        return data_transform