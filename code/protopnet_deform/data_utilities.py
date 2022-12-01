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



# Function: Apply normalisation to a given input function
def preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    assert x.size(1) == 3

    y = torch.zeros_like(x)

    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]

    return y



# Function: Allocate new tensor like x and apply the normalization used in the pretrained model
def preprocess_input_function(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    return preprocess(x=x, mean=mean, std=std)



# Function: Denormalise input(s), i.e., in this case images
def undo_preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    
    assert x.size(1) == 3
    
    y = torch.zeros_like(x)
    
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    
    return y



# Function: Allocate new tensor like x and undo the normalization used in the pretrained model
def undo_preprocess_input_function(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    
    return undo_preprocess(x=x, mean=mean, std=std)



# Function: Save preprocessed image(s)
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    
    # Make a copy of the image
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    
    # Revert any type of preprocessing
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    
    # FIXME: Erase after review
    # print('image index {0} in batch'.format(index))
    
    # Get unprocessed version of the image
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])

    # Save image    
    plt.imsave(fname, undo_preprocessed_img)


    return undo_preprocessed_img



# Function: Save image prototypes
def save_prototype(fname, load_img_dir, index):
    
    # Open an image file and save it
    p_img = plt.imread(os.path.join(load_img_dir, 'prototype-img'+str(index)+'.png'))
    plt.imsave(fname, p_img)

    return
    


# Function: Save prototype bbox
def save_prototype_box(fname, load_img_dir, index):
    
    # Open an image file and save it
    p_img = plt.imread(os.path.join(load_img_dir, 'prototype-img-with_box'+str(index)+'.png'))
    plt.imsave(fname, p_img)

    return
    


# Function: Save image with a bounding-box
def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    
    # Convert RGB to BGR with OpenCV
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    
    # Draw rectangle around bbox
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    
    # Convert into RGB and type float
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    
    # Save image
    plt.imsave(fname, img_rgb_float)


    return



# Function: Save deformable prototype information
def save_deform_info(model, offsets, input, activations, save_dir, prototype_img_filename_prefix, proto_index, prototype_layer_stride=1):
    
    # Get prototype shape
    prototype_shape = model.prototype_shape

    # FIXME: Review this piece of code
    if not hasattr(model, "prototype_dilation"):
        dilation = model.prototype_dillation
    else:
        dilation = model.prototype_dilation
    
    
    # Get input shape
    original_img_size = input.shape[0]

    colors = [
        (230/255, 25/255, 75/255),
        (60/255, 180/255, 75/255),
        (255/255, 225/255, 25/255),
        (0, 130/255, 200/255),
        (245/255, 130/255, 48/255),
        (70/255, 240/255, 240/255),
        (240/255, 50/255, 230/255),
        (170/255, 110/255, 40/255), (0,0,0)
    ]


    # Get Prototype Activations
    argmax_proto_act_j = list(np.unravel_index(np.argmax(activations, axis=None), activations.shape))
    fmap_height_start_index = argmax_proto_act_j[0] * prototype_layer_stride
    fmap_width_start_index = argmax_proto_act_j[1] * prototype_layer_stride


    # Get original image(s) with bboxes
    original_img_j_with_boxes = input.copy()


    # Get number of deformable groups (model.num_prototypes // model.num_classes)
    num_def_groups = 1

    # Get deformable group(s) indices and offset
    def_grp_index = proto_index % num_def_groups
    def_grp_offset = def_grp_index * 2 * prototype_shape[-2] * prototype_shape[-1]


    # Iterate through prototype shape(s) to get the prototypes
    for i in range(prototype_shape[-2]):
        for k in range(prototype_shape[-1]):
            
            # Offsets go in order height offset, width offset
            h_index = def_grp_offset + 2 * (k + prototype_shape[-2]*i)
            w_index = h_index + 1
            h_offset = offsets[0, h_index, fmap_height_start_index, fmap_width_start_index]
            w_offset = offsets[0, w_index, fmap_height_start_index, fmap_width_start_index]

            # Subtract prototype_shape // 2 because fmap start indices give the center location, and we start with top left
            def_latent_space_row = fmap_height_start_index + h_offset + (i - prototype_shape[-2] // 2) * dilation[0]
            def_latent_space_col = fmap_width_start_index + w_offset + (k - prototype_shape[-1] // 2)* dilation[1]

            # Map the coordinates above into the image under analysis
            def_image_space_row_start = int(def_latent_space_row * original_img_size / activations.shape[-2])
            def_image_space_row_end = int((1 + def_latent_space_row) * original_img_size / activations.shape[-2])
            def_image_space_col_start = int(def_latent_space_col * original_img_size / activations.shape[-1])
            def_image_space_col_end = int((1 + def_latent_space_col) * original_img_size / activations.shape[-1])


            # Get the image with this bbox
            img_with_just_this_box = input.copy()
            
            # Draw rectangle 
            cv2.rectangle(
                img_with_just_this_box,
                (def_image_space_col_start, def_image_space_row_start),
                (def_image_space_col_end, def_image_space_row_end),
                colors[i*prototype_shape[-1] + k],
                1
            )


            # Save this image
            plt.imsave(
                os.path.join(save_dir, prototype_img_filename_prefix + str(proto_index) + '_patch_' + str(i*prototype_shape[-1] + k) + '-with_box.png'),
                img_with_just_this_box,
                vmin=0.0,
                vmax=1.0
            )


            # Draw rectangle ont the other image with bboxes
            cv2.rectangle(
                original_img_j_with_boxes,
                (def_image_space_col_start, def_image_space_row_start),
                (def_image_space_col_end, def_image_space_row_end),
                colors[i*prototype_shape[-1] + k],
                1
            )
            

            # Save this image uppon a given condition (defined by the original authors)
            if not (def_image_space_col_start < 0 or def_image_space_row_start < 0 or def_image_space_col_end >= input.shape[0] or def_image_space_row_end >= input.shape[1]):
                plt.imsave(
                    os.path.join(save_dir, prototype_img_filename_prefix + str(proto_index) + '_patch_' + str(i*prototype_shape[-1] + k) + '.png'),
                    input[def_image_space_row_start:def_image_space_row_end, def_image_space_col_start:def_image_space_col_end, :],
                    vmin=0.0,
                    vmax=1.0
                )


    # Save this image
    plt.imsave(
        os.path.join(save_dir, prototype_img_filename_prefix + str(proto_index) + '-with_box.png'),
        original_img_j_with_boxes,
        vmin=0.0,
        vmax=1.0
    )

    return



#  Dataset
# STANFORDCARSDataset: Dataset Class
class STANFORDCARSDataset(Dataset):
    def __init__(self, data_path, cars_subset, augmented, cropped=True, transform=None):
        
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
        # image_fnames = image_fnames
        image_labels = np.array(image_labels) - 1
        # self.image_bboxes = image_bboxes


        # Get the right path and labels of the augmented version of the dataset
        images_fpaths, images_flabels = list(), list()


        # Create labels dictionary
        labels_dict = dict()

            
        # Go to folder
        for fname, label in zip(image_fnames, image_labels):
            
            # Enter the path of images
            if augmented:
                image_folder_path = os.path.join(self.images_path, fname.split('.')[0], "augmented")
            else:
                image_folder_path = os.path.join(self.images_path, fname.split('.')[0])




            # Get images in this folder
            images = [i for i in os.listdir(image_folder_path) if not i.startswith('.')]

            # Clean directories (if needed)
            images = [path for path in images if not os.path.isdir(os.path.join(image_folder_path, path))]


            # Add to labels dictionary
            labels_dict[fname.split('.')[0]] = label


            # Go through these images
            for img in images:

                # Get image path
                img_path = os.path.join(image_folder_path, img)

                # Append path and label to the proper list
                images_fpaths.append(img_path)
                images_flabels.append(label)



        # Extra variables
        self.images_fpaths = images_fpaths
        self.images_flabels = images_flabels
        self.class_names = sio.loadmat(os.path.join(data_path, "stanfordcars", "car_devkit", "devkit", "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        self.idx_to_class = {i:cls for i, cls in enumerate(self.class_names)}
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
        # Open the file
        image = Image.open(self.images_fpaths[idx]).convert('RGB')

        # Get labels
        label = self.images_flabels[idx]


        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label



# CUB2002011Dataset: Dataset Class
class CUB2002011Dataset(Dataset):
    def __init__(self, data_path, classes_txt, augmented, transform=None):

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
            if augmented:
                img_fnames = [i for i in os.listdir(os.path.join(data_path, folder, "augmented")) if not i.startswith('.')]
            else:
                img_fnames = [i for i in os.listdir(os.path.join(data_path, folder)) if not i.startswith('.')]


            # Get each image
            for img_name in img_fnames:

                # Build the complete path
                if augmented:
                    img_path = os.path.join(folder, "augmented", img_name)
                else:
                    img_path = os.path.join(folder, img_name)


                # Append this path to our variable of images_fpaths
                images_fpaths.append(img_path)
            

        # Clean images_fpaths (to prevent IsADirectoryError errors)
        images_fpaths = [path for path in images_fpaths if not os.path.isdir(os.path.join(data_path, path))]
        

        # Add this to our variables
        self.data_path = data_path
        self.images_fpaths = images_fpaths


        # Extract labels from data path
        labels = np.genfromtxt(classes_txt, dtype=str)
        labels_dict = dict()
        for label_info in labels:
            labels_dict[label_info[1]] = int(label_info[0]) - 1


        # print(f"Number of Labels: {len(labels_dict)}")
        # print(f"Labels dict: {labels_dict}")

        self.labels_dict = labels_dict


        # Transforms
        self.transform = transform

        # Augmented
        self.augmented = augmented


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
    def __init__(self, data_path, subset, cropped, augmented, transform=None):

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

                # Check augmented files
                if augmented:
                    augmented_files = [i for i in os.listdir(os.path.join(self.images_dir, img_name, "augmented"))]
                    augmented_files = [i for i in augmented_files if not i.startswith('.')]


                    # Iterate through these files
                    for aug_img in augmented_files:
                        ph2_dataset_imgs.append(aug_img)
                        ph2_dataset_labels.append(img_label)        

                else:
                    ph2_dataset_imgs.append(img_name)
                    ph2_dataset_labels.append(img_label)


        # Create final variables
        self.images_names = ph2_dataset_imgs.copy()
        self.images_labels = ph2_dataset_labels.copy()
        self.cropped = cropped
        self.augmented = augmented


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
        if self.cropped:

            if self.augmented:
                img_name = self.images_names[idx].split('_')[0]
                img = self.images_names[idx]
                img_path = os.path.join(self.images_dir, f"{img_name}", "augmented", img)
            
            else:
                img_name = self.images_names[idx]
                img_path = os.path.join(self.images_dir, img_name, f"{img_name}.png")
        
        else:
            img_path = os.path.join(self.images_dir, self.images_names[idx], f"{self.images_names[idx]}_Dermoscopic_Image", f"{self.images_names[idx]}.bmp")
        
        
        # Open image
        image = Image.open(img_path).convert('RGB')

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label
