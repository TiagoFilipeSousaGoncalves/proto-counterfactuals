import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.draw as drw
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# Function: Pre-processing function
def preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y



# Function: Apply pre-processing function
def preprocess_input_function(x):
    return preprocess(x)



# Function: Undo pre-processing
def undo_preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y



# Function: Undo pre-processing
def undo_preprocess_input_function(x):
    return undo_preprocess(x)



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



# Function: Save preprocessed image
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    # print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img



# Function: Save prototype
def save_prototype(prototype_fname, prototypes_img_dir, index):
    p_img = plt.imread(os.path.join(prototypes_img_dir, 'prototype-img'+str(index)+'.png'))
    plt.imsave(prototype_fname, p_img)
    return



# Function: Save prototype self activation
def save_prototype_self_activation(prototype_self_act_fname, prototypes_img_dir, index):
    p_img = plt.imread(os.path.join(prototypes_img_dir, 'prototype-img-original_with_self_act'+str(index)+'.png'))
    plt.imsave(prototype_self_act_fname, p_img)
    return



# Function: Save prototype with bounding box
def save_prototype_original_img_with_bbox(
        prototype_bbox_fname,
        prototypes_img_dir,
        index,
        bbox_height_start,
        bbox_height_end,
        bbox_width_start,
        bbox_width_end,
        color=(0, 255, 255)):

    p_img_bgr = cv2.imread(os.path.join(prototypes_img_dir, 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(prototype_bbox_fname, p_img_rgb)
    return



# Function: Save image with bounding box
def imsave_with_bbox(
        high_act_patch_img_bbox_fname,
        img_rgb,
        bbox_height_start,
        bbox_height_end,
        bbox_width_start,
        bbox_width_end,
        color=(0, 255, 255)):

    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(high_act_patch_img_bbox_fname, img_rgb_float)
    return



# CUB2002011Dataset: Dataset Class
class CUB2002011Dataset(Dataset):
    def __init__(self, data_path, fold=0, split="train", transform=None):

        """
        Args:
            data_path (string): Data directory.
            classes_txt (string): File with the classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        assert split in ("train", "val", "test")
        self.fold = fold
        self.split = split
        self.data_path = data_path


        # Get the path of the split
        splits_df = pd.read_csv(os.path.join(data_path, "processed", "data_splits.csv"))
        dataset = splits_df[(splits_df['fold'] == fold) & (splits_df['split'] == split)]

        # Extract labels from data path
        labels = np.genfromtxt(os.path.join(data_path, "CUB_200_2011", "classes.txt"), dtype=str)
        labels_dict = dict()
        for label_info in labels:
            labels_dict[label_info[1]] = int(label_info[0]) - 1


        # Extract bounding box information
        images = np.genfromtxt(os.path.join(data_path, "CUB_200_2011", "images.txt"), dtype=str)
        bounding_boxes = np.genfromtxt(os.path.join(data_path, "CUB_200_2011", "bounding_boxes.txt"), dtype=str)

        # Create a list with all info
        bbox_dict = dict()
        for img in images:
            for bbox in bounding_boxes:
                img_id = img[0]
                img_fname = img[1]
                bbox_id = bbox[0]
                bbox_coord = bbox[1::]

                if img_id == bbox_id:
                    img_fname_ = img_fname.split("/")[-1]
                    bbox_dict[img_fname_] = bbox_coord


        self.dataset = dataset
        self.labels_dict = labels_dict
        self.bbox_dict = bbox_dict
        self.transform = transform

        return



    # Method: __len__
    def __len__(self):
        return len(self.dataset)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # Get image
        img_path = self.dataset.iloc[idx]['image_fnames']

        # Load image
        image = Image.open(os.path.join(self.data_path, "CUB_200_2011", "images", img_path)).convert('RGB')

        # Load bounding box
        img_bbox = self.bbox_dict[img_path.split('/')[-1]]


        # Get bounding boxes info
        x = float(img_bbox[0])
        y = float(img_bbox[1])
        width = float(img_bbox[2])
        height = float(img_bbox[3])

        # Crop image
        image = image.crop((x, y, x+width, y+height))


        # Get labels
        folder = img_path.split("/")[0]
        label = self.labels_dict[folder]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



# PH2Dataset: Dataset Class
class PH2Dataset(Dataset):
    def __init__(self, data_path, fold=0, split="train", transform=None):

        """
        Args:
            data_path (string): Data directory.
            fold (int): Fold number.
            split (string): Data split ('train', 'val', 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        assert split in ("train", "val", "test"), "Subset must be in ('train', 'val', 'test')."

        # Get the path of the split
        splits_df = pd.read_csv(os.path.join(data_path, "processed", "data_splits.csv"))
        dataset = splits_df[(splits_df['fold'] == fold) & (splits_df['split'] == split)]

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


        # Labels Dictionary
        labels_dict = dict()
        for img, label in zip(dataset["image_name"].values,dataset["image_label"].values):
            labels_dict[img] = label


        self.data_path = data_path
        self.dataset = dataset
        self.labels_dict = labels_dict
        self.transform = transform


        return



    # Method: __len__
    def __len__(self):
        return len(self.dataset)



    # Method: __getitem__
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load and crop image
        image_name = self.dataset.iloc[idx]['image_name']
        img_path = os.path.join(self.data_path, "images", image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp")
        image = Image.open(img_path).convert('RGB')
        image = self.crop_image(image=image, image_name=image_name)

        # Get labels
        label = self.dataset.iloc[idx]["image_label"]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label


    # Method: Crop image
    def crop_image(self, image, image_name):

        # Open mask
        pil_mask = Image.open(os.path.join(self.data_path, "images", image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp")).convert('L')

        # Convert PIL mask to NumPy array
        npy_mask = np.array(pil_mask.copy())

        # We have to be sure we are working with binary image
        ret, binary = cv2.threshold(npy_mask, 127, 255, cv2.THRESH_BINARY)

        # Get countours of this image
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding box from mask
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop image
        crop_img = image.copy().crop((x, y, x+w, y+h))

        return crop_img



# PAPILADataset: Dataset Class
class PAPILADataset(Dataset):
    def __init__(self, data_path, fold=0, split='train', transform=None):

        """
        Args:
            data_path (string): Data directory.
            fold (int): Fold number.
            split (string): Data split ('train', 'val', 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        assert split in ("train", "val", "test"), "Split must be in ('train', 'val', 'test')."

        # Get the path of the split
        splits_df = pd.read_csv(os.path.join(data_path, "processed", "data_splits.csv"))
        dataset = splits_df[(splits_df['fold'] == fold) & (splits_df['split'] == split)]

        # Labels Dictionary
        labels_dict = dict()
        for img, label in zip(dataset["images_fnames"].values, dataset["images_labels"].values):
            labels_dict[img] = label


        self.data_path = data_path
        self.dataset = dataset
        self.images_labels = dataset["images_labels"].values
        self.labels_dict = labels_dict
        self.transform = transform


        return



    # Method: __len__
    def __len__(self):
        return len(self.dataset)



    # Method: __getitem__
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load and crop image
        image_name = self.dataset.iloc[idx]['images_fnames'].replace(".png", ".jpg")
        image = Image.open(os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "FundusImages", image_name))
        image = self.crop_image(image=image, image_name=image_name)

        # Get labels
        label = self.dataset.iloc[idx]['images_labels']

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label


    # Method: Crop image
    def crop_image(self, image, image_name):

        # Convert to NumPy
        npy_img = np.array(image)

        # Get ROIs (We use both expert annotations)
        cup_path1 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_cup_exp1.txt")
        disc_path1 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_disc_exp1.txt")
        cup_path2 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_cup_exp2.txt")
        disc_path2 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_disc_exp2.txt")


        # Generate ROIs
        cup_mask1 = self.contour_to_mask(cup_path1, npy_img.shape)
        disc_mask1 = self.contour_to_mask(disc_path1, npy_img.shape)
        cup_mask2 = self.contour_to_mask(cup_path2, npy_img.shape)
        disc_mask2 = self.contour_to_mask(disc_path2, npy_img.shape)

        # Generate final mask
        final_mask = np.zeros_like(cup_mask1)
        final_mask[cup_mask1==1] = 1
        final_mask[disc_mask1==1] = 1
        final_mask[cup_mask2==1] = 1
        final_mask[disc_mask2==1] = 1
        # final_mask *= 255

        # Some debug operations
        # image = apply_mask(image=npy_img, mask=final_mask, color=(1.0, 1.0, 1.0))
        # plt.imshow(image, cmap="gray")
        # plt.show()


        # We have to be sure we are working with binary image
        ret, binary = cv2.threshold(final_mask*255, 127, 255, cv2.THRESH_BINARY)
        # plt.imshow(binary, cmap="gray")
        # plt.show()

        # Get countours of this image
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding box from mask
        x, y, w, h = cv2.boundingRect(contours[0])


        # Crop image
        crop_img = image.crop((x, y, x+w, y+h))
        # plt.imshow(np.array(crop_img), cmap="gray")
        # plt.show()


        # Some debug tests using OpenCV
        # crop_img = npy_img[y:y+h, x:x+w]
        # plt.imshow(np.array(crop_img), cmap="gray")
        # plt.show()

        return crop_img

    # Function: Return mask given a contour and the shape of image
    # contour_to_mask() and apply_mask() functions taken from https://github.com/matterport/Mask_RCNN
    def contour_to_mask(self, contour_txt_path, img_shape):
        """
        Return mask given a contour and the shape of image
        """

        # Get coordinates
        c = np.loadtxt(contour_txt_path)

        # Generate mask
        mask = np.zeros(img_shape[:-1], dtype=np.uint8)
        rr, cc = drw.polygon(c[:,1], c[:,0])
        mask[rr, cc] = 1

        return mask



    # Function: Apply the given mask to the image
    def apply_mask(self, image, mask, color, alpha=0.5):
        """
        Apply the given mask to the image.
        """


        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])


        return image
