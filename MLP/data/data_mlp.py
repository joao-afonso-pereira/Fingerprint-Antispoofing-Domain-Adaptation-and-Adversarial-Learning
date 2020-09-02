import torch
from torch.utils.data import Dataset
import scipy.io as sio
import os
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from torchvision import transforms
import pandas as pd
import sys

class numpyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()
    
#PATH = "L:/FPAD/Dataset/LivDet2015/train/"
#DATASET="CrossMatch"
BATCH_SIZE = 64

def get_data_loaders(path, dataset, test_material, croped=False, unseen_attack=False):
    
    if unseen_attack==True:
        
        if dataset == "CrossMatch":
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]
            real_train = np.array(range(1000))
            real_test = np.array(range(1000, 1500))
        elif dataset == "Digital_Persona":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.array(range(750))
            real_test = np.array(range(750, 1000))
        elif dataset == "GreenBit":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.array(range(750))
            real_test = np.array(range(750, 997))
        elif dataset == "Hi_Scan":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.array(range(750))
            real_test = np.array(range(750, 1000))
        elif dataset == "Time_Series":
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]
            real_train = np.array(range(2960))
            real_test = np.array(range(2960, 4440))
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
        elif dataset == "Digital_Persona":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.random.randint(low=0, high=1000, size=250)
        elif dataset == "GreenBit":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.random.randint(low=0, high=997, size=250)
        elif dataset == "Hi_Scan":
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]
            real_train = np.random.randint(low=0, high=1000, size=250)
        elif dataset == "Time_Series":
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]
            real_train = np.random.randint(low=0, high=4440, size=1480)
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

class FPAD(Dataset):
    def __init__(self,
                 path,
                 dataset,
                 material_idx,
                 real_idx,
                 croped = False):

        self.material_idx = material_idx
        self.real_idx = real_idx
        self.dataset = dataset
        
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
                self.fake_dir = path + self.dataset + "_" + self.materials[index] + ".csv"
            if croped == True:
                self.fake_dir = path + self.dataset + "_" + self.materials[index] + "_c.csv"
            
            df = pd.read_csv(self.fake_dir)
    
            features_pa = df.to_numpy(copy=True)
            features_pa = features_pa[:, 1:]
            
            n_pa = features_pa.shape[0]
    
            count = count + n_pa
    
            X.extend(features_pa)
            y.extend([1]*n_pa)
            f.extend([index]*n_pa)
            f_norm.extend([index]*n_pa)
    
        self.n_presentation_attack_samples = count

        #BONAFIDE SAMPLES
        
        if croped == False:
            self.real_dir = path + self.dataset + "_real.csv"
        else:
            self.real_dir = path + self.dataset + "_real_c.csv"
            
        df = pd.read_csv(self.real_dir)
        
        features_bf = df.to_numpy(copy=True)
        features_bf = features_bf[:, 1:]
        
        self.n_bonafide_samples = self.real_idx.shape[0]
        
        features_bf = features_bf[self.real_idx]        

        # append real_data to X, y, and f arrays
        X.extend(features_bf)
        y.extend([0]*self.n_bonafide_samples)
        f.extend([-1]*self.n_bonafide_samples)
        f_norm.extend([-1]*self.n_bonafide_samples)

        self.X = np.array(X)
        self.y = np.array(y)
        self.f = np.array(f)
        self.f_norm = np.array(f_norm)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        tranformation = self.tranformations()
        return tranformation(self.X[index]), self.y[index], self.f[index], self.f_norm[index]
    
    def tranformations(self):
        data_transform = transforms.Compose([numpyToTensor()])
        return data_transform
    
#%%
    
if __name__ == '__main__':
    
    BATCH_SIZE = 32
    DATASET = "CrossMatch"

    data = FPAD(DATASET, material_idx=[1, 2], real_idx=np.array(range(1000)), croped=False)
    data_test = FPAD(DATASET, material_idx=[0], real_idx=np.array(range(1000, 1500)), croped=False)
    
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}
  
  
    train_loader = torch.utils.data.DataLoader(data_train, **params)
    valid_loader = torch.utils.data.DataLoader(data_val, **params)
    test_loader = torch.utils.data.DataLoader(data_test, **params)

