# Imports
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
import cv2
import skimage.draw as drw

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
                    bbox_dict[img_fname] = bbox_coord


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
    def __init__(self, data_path, subset, cropped=True, augmented=False, transform=None):

        """
        Args:
            data_path (string): Data directory.
            cropped (boolean): If we want the cropped version of the data set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        assert subset in ("train", "val", "test"), "Subset must be in ('train', 'val', 'test')."
        if augmented:
            assert subset == 'train'

        # Select if you want the cropped version or not
        if cropped:
            self.images_dir = os.path.join(data_path, "processed", "images", subset, "cropped")
        else:
            self.images_dir = os.path.join(data_path, "processed", "images", subset, "raw")


        # Get image names
        image_names = [i for i in os.listdir(self.images_dir) if not i.startswith('.')]
        image_names = [i.split('.')[0] for i in image_names]



        # Get labels
        ph2_xlsx = os.path.join(data_path, "metadata", "PH2_dataset.xlsx")

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
                        ph2_dataset_imgs.append(os.path.join(self.images_dir, img_name, 'augmented', aug_img))
                        ph2_dataset_labels.append(img_label)

                ph2_dataset_imgs.append(os.path.join(self.images_dir, img_name, f"{img_name}.png"))
                ph2_dataset_labels.append(img_label)


        # Create final variables
        self.images_names = ph2_dataset_imgs.copy()
        self.images_labels = ph2_dataset_labels.copy()
        self.cropped = cropped
        self.augmented = augmented


        # Labels Dictionary
        labels_dict = dict()
        for img, label in zip(ph2_dataset_imgs.copy(), ph2_dataset_labels.copy()):
            labels_dict[img] = label

        self.labels_dict = labels_dict


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
        img_path = self.images_names[idx]

        # Open image
        image = Image.open(img_path).convert('RGB')

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



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
        for img, label in zip(dataset["images_fnames"].values,dataset["images_labels"].values):
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
        image_name = self.dataset.iloc[idx]['images_fnames']
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
