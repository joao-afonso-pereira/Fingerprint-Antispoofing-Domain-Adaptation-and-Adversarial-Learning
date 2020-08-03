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
          
#%% SPECIFICATIONS -------------------------------------------------------------------------------------------------

BATCH_SIZE = 12 

#Image dimensions
MIN_CROP_WIDTH = -1
MIN_CROP_HEIGHT = -1

#%% Get Data Loaders -------------------------------------------------------------------------------------------------
#
#  Input: 
#           - [STR] path (of the images)
#           - [STR] dataset (CrossMatch, Digital_Persona, GreenBit, Hi_Scan, Time_Series)
#           - [INT] test_material (material id to use for testing - 0, 1 or 2 for CrossMatch and Time_Series | 0, 1, 2 or 3 for the remaining datasets)
#           - [BOOL] croped (True to use the previously segmented and cropped images)
#           - [BOOL] unseen_attack (True to create a test dataset with an unseen attack using the test material)
#
#  Output: 
#           - train, validation and test data loaders

def get_data_loaders(path, dataset, test_material, croped=True, unseen_attack=False):
    
    global MIN_CROP_WIDTH 
    global MIN_CROP_HEIGHT 
    global BATCH_SIZE

    sensor = dataset
    
    if unseen_attack==True:
        
        if dataset == "CrossMatch":

            #attack materials idxs and names
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]

            #real image idxs in the dataset
            real_train = np.array(range(1000))
            real_test = np.array(range(1000, 1500))

            #dimensions
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 247
            
        elif dataset == "Digital_Persona":

            #attack materials idxs and names
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

            #real image idxs in the dataset
            real_train = np.array(range(750))
            real_test = np.array(range(750, 1000))

            #dimensions
            MIN_CROP_WIDTH = 224
            MIN_CROP_HEIGHT = 235

        elif dataset == "GreenBit":

            #attack materials idxs and names
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

            #real image idxs in the dataset
            real_train = np.array(range(750))
            real_test = np.array(range(750, 997))

            #dimensions
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 271

        elif dataset == "Hi_Scan":

            BATCH_SIZE = 10 #smaller batch size because of memory issues

            #attack materials idxs and names
            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

            #real image idxs in the dataset
            real_train = np.array(range(750))
            real_test = np.array(range(750, 1000))

            #dimensions
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 311

        elif dataset == "Time_Series":

            #attack materials idxs and names
            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]

            #real image idxs in the dataset
            real_train = np.array(range(2960))
            real_test = np.array(range(2960, 4440))

            #dimensions
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 225

        else:
            sys.exit("Error: incorrect dataset!")
  
                   
        train_materials = np.delete(materials_list, test_material) #train_materials = materials - test_material (unseen-attack)
        
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

            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 247

        elif dataset == "Digital_Persona":

            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

            real_train = np.random.randint(low=0, high=1000, size=250)

            MIN_CROP_WIDTH = 224
            MIN_CROP_HEIGHT = 235

        elif dataset == "GreenBit":

            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

            real_train = np.random.randint(low=0, high=997, size=250)
  
            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 271

        elif dataset == "Hi_Scan":

            BATCH_SIZE = 10

            materials_list = [0,1,2,3]
            materials_name = ["Ecoflex_00_50", "WoodGlue", "Gelatine", "Latex"]

            real_train = np.random.randint(low=0, high=1000, size=250)

            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 311

        elif dataset == "Time_Series":

            materials_list = [0,1,2]
            materials_name = ["Body_Double", "Ecoflex", "Playdoh"]

            real_train = np.random.randint(low=0, high=4440, size=1480)

            MIN_CROP_WIDTH = 225
            MIN_CROP_HEIGHT = 225

        else:
            sys.exit("Error: incorrect dataset!")        
        
        
        train_materials = [test_material] #one-attack = just one material to train and to test
        
        dataset = FPAD(path, dataset, material_idx=train_materials, real_idx=real_train, croped=croped)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _dataset, data_test = torch.utils.data.random_split(dataset, [train_size, test_size])
    
        train_size = int(0.8 * len(_dataset))
        val_size = len(_dataset) - train_size
        data_train, data_val = torch.utils.data.random_split(_dataset, [train_size, val_size])
      
    #Print info about datasets
    print('\n--------------------------------------')
    print('Dataset: ' + sensor)
    print('Train materials: ', end="")
    for material in train_materials:
        print(materials_name[material], end="  ")
    print('\nTest material: {}'.format(materials_name[test_material]))
    
    #Params of the data loaders
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0} 
  
    #Data loaders
    train_loader = torch.utils.data.DataLoader(data_train, **params)
    valid_loader = torch.utils.data.DataLoader(data_val, **params)
    test_loader = torch.utils.data.DataLoader(data_test, **params)

    #Print info about data loaders
    print('\nDatasets size: Train {}, Val {}, Test {}'.format(len(data_train),
                                                           len(data_val),
                                                           len(data_test)))
  
    return train_loader, valid_loader, test_loader

#%% DATASET -------------------------------------------------------------------------------------------------------

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
        
        index_norm = 0
    
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
            f_norm.extend([index_norm]*len(fake_names))
            
            index_norm = index_norm + 1
    
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
        
        
        if self.croped == True:    
            
            
            sample = Image.fromarray(np.uint8(sample))
            width, height = sample.size
        
            
            left = int((width-MIN_CROP_WIDTH)/2)
            right = width - MIN_CROP_WIDTH - left
            top = int((height-MIN_CROP_HEIGHT)/2)
            bottom = height - MIN_CROP_HEIGHT - top
            
            sample = ImageOps.crop(sample, (left, top, right, bottom))            
            
        transformation = self.transformations()
        
        sample = np.array(sample)
        
        
        
        if self.dataset == "Digital_Persona":
            
            sample = sample[:,:,0]
            sample =  np.transpose(sample)
              
    
        sample.reshape((1, sample.shape[0], sample.shape[1]))
       
        return transformation(sample).view((1, sample.shape[0], sample.shape[1])), self.y[idx], self.f[idx], self.f_norm[idx]
    
    def transformations(self):
        data_transform = transforms.Compose([transforms.ToTensor()])
        return data_transform
    
#%% MAIN
    
if __name__ == '__main__':
    
    train, val, test = get_data_loaders("/ctm-hdd-pool01/DB/LivDet2015/train/", "Digital_Persona", 0, croped=True, unseen_attack=True)

    for i, (x, y, f, _) in enumerate(train):
        print(x.shape)
        print(y.shape)
        print(f.shape)
        break