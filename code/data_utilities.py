# Imports
import os
import _pickle as cPickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# Sklearn Imports
from sklearn.model_selection import train_test_split

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# General
# Function: Resize images
def resize_images(datapath, newpath, newheight=512):
    
    # Create new directories (if necessary)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    

    # Go through data directory and generate new (resized) images
    for f in tqdm(os.listdir(datapath)):
        if(f.endswith(".jpg") or f.endswith('.png')):
            img = Image.open(os.path.join(datapath, f))
            w, h = img.size
            ratio = w / h
            new_w = int(np.ceil(newheight * ratio))
            new_img = img.resize((new_w, newheight), Image.ANTIALIAS)
            new_img.save(os.path.join(newpath, f))


    return



# Function: Preprocess base function 
def preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    assert x.size(1) == 3
    
    y = torch.zeros_like(x)
    
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    
    
    return y



# Function: Function to process inputs
def preprocess_input_function(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x=x, mean=mean, std=std)



# Funtion: Revert processing function
def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    
    return y



# Function: Apply revert processing function
def undo_preprocess_input_function(x, mean, std):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    
    return undo_preprocess(x, mean=mean, std=std)



# CUB2002011Dataset: Dataset Class
class CUB2002011Dataset(Dataset):
    def __init__(self, data_path, classes_txt, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        

        # Data path (get images folders)
        images_folders = [f for f in os.listdir(data_path) if not f.startswith('.')]

        # Enter each folder and add the image path to our images_fpaths variable
        images_fpaths = list()
        for folder in images_folders:

            # Get images
            img_fnames = [i for i in os.listdir(os.path.join(data_path, folder)) if not i.startswith('.')]

            # Get each image
            for img_name in img_fnames:

                # Build the complete path
                img_path = os.path.join(folder, img_name)

                # Append this path to our variable of images_fpaths
                images_fpaths.append(img_path)
        

        # Add this to our variables
        self.data_path = data_path
        self.images_fpaths = images_fpaths

        
        # Extract labels from data path
        labels = np.genfromtxt(classes_txt, dtype=str)
        labels_dict = dict()
        for label_info in labels:
            labels_dict[label_info[1]] = int(label_info[0]) -1
        
        # print(f"Number of Labels: {len(labels_dict)}")
        # print(f"Labels dict: {labels_dict}")

        self.labels_dict = labels_dict
        

        # Transforms
        self.transform = transform


        return



    # Method: __len__
    def __len__(self):
        
        return len(self.images_fpaths)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_path = self.images_fpaths[idx]
        image = Image.open(os.path.join(self.data_path, img_path)).convert('RGB')

        # Get labels
        folder = img_path.split("/")[0]
        label = self.labels_dict[folder]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



# Run this file to test its functions
if __name__ == "__main__":

    data_path = os.path.join("data", "cub_200_2011", "processed_data", "train", "images")
    classes_txt = os.path.join("data", "cub_200_2011", "source_data", "classes.txt")
    dataset = CUB2002011Dataset(data_path=data_path, classes_txt=classes_txt)
    print(f"Length of the dataset: {len(dataset)}")