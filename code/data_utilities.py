# Imports
import os
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset



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



# Function: Save preprocessed image
def save_preprocessed_img(fname, preprocessed_imgs, mean, std, index=0):
    
    # Create copy of the image
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(x=img_copy, mean=mean, std=std)
    
    print('Image index {0} in batch'.format(index))
    
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    
    return undo_preprocessed_img



# Function: Save prototype
def save_prototype(fname, load_img_dir, epoch, index):
    
    # Get prototype image
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
    
    # plt.axis('off')
    plt.imsave(fname, p_img)

    return



# Function: Save prototype self-activation
def save_prototype_self_activation(fname, load_img_dir, epoch, index):

    # Get prototype self-activation
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original_with_self_act'+str(index)+'.png'))
    
    # plt.axis('off')
    plt.imsave(fname, p_img)

    return



# Function: Save prototype image with bounding-box
def save_prototype_original_img_with_bbox(fname, load_img_dir, epoch, index, bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    
    # Load image with OpenCV
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    
    # Draw bounding-boc
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    
    # Get RGB image
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    # plt.imshow(p_img_rgb)
    # plt.axis('off')
    plt.imsave(fname, p_img_rgb)

    return



# Function: Save image with bounding-box
def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    
    # Load image with OpenCV
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    
    # Draw bounding-box
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    
    # Convert to RGB
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    # plt.imshow(img_rgb_float)
    # plt.axis('off')
    plt.imsave(fname, img_rgb_float)

    return



# Dataset
# STANFORDCARSDataset: Dataset Class
class STANFORDCARSDataset(Dataset):
    def __init__(self, data_path, cars_subset, cropped=True, transform=None):
        
        """
        Args:
            data_path (string): Data directory.
            cars_subset (string): Subset of data for training.
            cropped (Boolean): Specify if we want the cropped version fo the data set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        # Assert if cars_subset is available
        assert cars_subset in ("cars_train", "cars_test"), "Please provide a valid subset (cars_train, cars_test)."


        # Get the directory of images
        if cropped:
            self.images_path = os.path.join(data_path, "stanfordcars", cars_subset, "images_cropped")
        else:
            self.images_path = os.path.join(data_path, "stanfordcars", cars_subset, "images")


        # Get the correspondent .MAT file
        if cars_subset == "cars_train":
            mat_file = sio.loadmat(os.path.join(data_path, "stanfordcars", "car_devkit", "devkit", f"{cars_subset}_annos.mat"))
        else:
            mat_file = sio.loadmat(os.path.join(data_path, "stanfordcars", "car_devkit", "devkit", f"{cars_subset}_annos_withlabels.mat"))


        # Create lists to append stuff
        image_fnames = list()
        image_labels = list()
        image_bboxes = list()

        # Go through data points
        for entries in mat_file['annotations']:
            for sample in entries:
                # dtype=[('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('fname', 'O')])}
                bbox_x1 = sample[0][0,0]
                bbox_y1 = sample[1][0,0]
                bbox_x2 = sample[2][0,0]
                bbox_y2 = sample[3][0,0]
                label = sample[4][0,0]
                fname = sample[5][0]

                # Append fname
                image_fnames.append(fname)

                # Append label
                image_labels.append(label)

                # Append bbox
                image_bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))



        # Add these variables to the class
        self.image_fnames = image_fnames
        self.image_labels = np.array(image_labels) - 1
        self.image_bboxes = image_bboxes


        # Extra variables
        self.class_names = sio.loadmat(os.path.join(data_path, "stanfordcars", "car_devkit", "devkit", "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        self.idx_to_class = {i:cls for i, cls in enumerate(self.class_names)}


        # Transforms
        self.transform = transform


        return



    # Method: __len__
    def __len__(self):
        
        return len(self.image_fnames)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        # Open the file
        image = Image.open(os.path.join(self.images_path, self.image_fnames[idx])).convert('RGB')

        # Get labels
        label = self.image_labels[idx]


        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label



# CUB2002011Dataset: Dataset Class
class CUB2002011Dataset(Dataset):
    def __init__(self, data_path, classes_txt, transform=None):

        """
        Args:
            data_path (string): Data directory.
            classes_txt (string): File with the classes.
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



# PH2Dataset: Dataset Class
class PH2Dataset(Dataset):
    def __init__(self, data_path, subset, cropped, transform=None):

        """
        Args:
            data_path (string): Data directory.
            cropped (boolean): If we want the cropped version of the data set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        assert subset in ("train", "test"), "Subset must be in ('train', 'test')."


        # Directories
        ph2_dir = os.path.join(data_path, "ph2")


        # Select if you want the cropped version or not
        if cropped:
            self.images_dir = os.path.join(ph2_dir, "processed_images", subset, "cropped")
        else:
            self.images_dir = os.path.join(ph2_dir, "processed_images", subset, "raw")


        # Get image names
        image_names = [i for i in os.listdir(self.images_dir) if not i.startswith('.')]
        image_names = [i.split('.')[0] for i in image_names]



        # Get labels
        ph2_xlsx = os.path.join(ph2_dir, "PH2_dataset.xlsx")

        # Open PH2 XLSX file
        ph2_df = pd.read_excel(ph2_xlsx, skiprows=[i for i in range(12)])

        # Get only classification columns
        ph2_df = ph2_df.copy()[['Image Name', 'Common Nevus', 'Atypical Nevus', 'Melanoma']]


        # Create a diagnosis dictionary
        self.diagnosis_dict = {
            "Common Nevus":0,
            "Atypical Nevus":1,
            "Melanoma":2
        }

        # Create an inverse diagnosis dictionary
        self.inv_diagnosis_dict = {
            0:"Common Nevus",
            1:"Atypical Nevus",
            2:"Melanoma"
        }

        # Add column "Label"
        ph2_df["Label"] = -1


        # Go through the DataFrame
        ph2_df = ph2_df.copy().reset_index()

        for index, row in ph2_df.iterrows():

            # Get values
            if row['Common Nevus'] == "X":
                ph2_df.iloc[index, -1] = self.diagnosis_dict['Common Nevus']

            elif row['Atypical Nevus'] == "X":
                ph2_df.iloc[index, -1] = self.diagnosis_dict['Atypical Nevus']

            elif row['Melanoma'] == "X":
                ph2_df.iloc[index, -1] = self.diagnosis_dict['Melanoma']


        # Get X, y
        X, y = ph2_df.copy()['Image Name'].values, ph2_df.copy()['Label'].values


        # Create a variable "image_labels"
        ph2_dataset_imgs, ph2_dataset_labels = list(), list()

        # Iterate through X and y
        for img_name, img_label in zip(X, y):

            # If it exists in our directory, append it to the dataset
            if img_name in image_names:
                ph2_dataset_imgs.append(img_name)
                ph2_dataset_labels.append(img_label)


        # Create final variables
        self.images_names = ph2_dataset_imgs.copy()
        self.images_labels = ph2_dataset_labels.copy()


        # Transforms
        self.transform = transform


        return



    # Method: __len__
    def __len__(self):

        return len(self.images_names)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # Get images
        img_path = os.path.join(self.images_dir, self.images_names[idx])
        image = Image.open(img_path).convert('RGB')

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label
