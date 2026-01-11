# Imports
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
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
    def __init__(self, data_path, fold, subset, cropped, augmented, transform=None):

        """
        Args:
            data_path (string): Data directory.
            cropped (boolean): If we want the cropped version of the data set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        assert subset in ("train", "val", "test"), "Subset must be in ('train', 'val', 'test')."

        if augmented:
            assert subset == "train"


        # Select if you want the cropped version or not
        if cropped:
            self.images_dir = os.path.join(data_path, "processed", f"kf_{fold}", "splits", subset)

        else:
            pass


        # Get image names
        image_names = [i for i in os.listdir(self.images_dir) if not i.startswith('.')]



        # Read diagnostic labels
        labels, _, patID = self.get_diagnosis(
            patient_data_od_path=os.path.join(data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ClinicalData", "patient_data_od.xlsx"),
            patient_data_os_path=os.path.join(data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ClinicalData", "patient_data_os.xlsx")
        )


        # Get X, y
        # Get the unique patient indices
        _, unique_indices, _, _ = np.unique(ar=patID,  return_index=True, return_inverse=True, return_counts=True)
        # print(unique_indices)


        # Get X and y
        X = patID[unique_indices]
        y = labels[unique_indices]
        # print(X, y)


        # Create temporary variables with image names and labels
        papila_images, papila_labels = list(), list()


        # Go through each patient ID
        for p_id, p_label in zip(X, y):

            # Get folder name for the image of the right eye
            right_img = "RET%03dOD" % p_id
            papila_images.append(right_img)
            papila_labels.append(p_label)


            # Get the folder name for the image of the left eye
            left_img = "RET%03dOS" % p_id
            papila_images.append(left_img)
            papila_labels.append(p_label)


        # Create the variables we will use to create the dataset
        papila_dataset_imgs, papila_dataset_labels = list(), list()

        # Iterate through X and y
        for img_name, img_label in zip(papila_images, papila_labels):

            # If it exists in our directory, append it to the dataset
            if img_name in image_names:

                # Check augmented files
                if augmented:
                    augmented_files = [i for i in os.listdir(os.path.join(self.images_dir, img_name, "augmented"))]
                    augmented_files = [i for i in augmented_files if not i.startswith('.')]


                    # Iterate through these files
                    for aug_img in augmented_files:
                        papila_dataset_imgs.append(os.path.join(self.images_dir, img_name, "augmented", aug_img))
                        papila_dataset_labels.append(img_label)

                else:
                    papila_dataset_imgs.append(os.path.join(self.images_dir, img_name, f"{img_name}.png"))
                    papila_dataset_labels.append(img_label)


        # Create final variables
        self.images_names = papila_dataset_imgs.copy()
        self.images_labels = papila_dataset_labels.copy()
        self.cropped = cropped
        self.augmented = augmented


        # Labels Dictionary
        labels_dict = dict()
        for img, label in zip(papila_dataset_imgs.copy(), papila_dataset_labels.copy()):
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
        if self.cropped:
            img_path = self.images_names[idx]
        else:
            pass


        # Open image
        image = Image.open(img_path).convert('RGB')

        # Get labels
        label = self.images_labels[idx]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



    # Function: Prepare the Data Frame to be readable
    def _fix_df(self, df):
        """
        Prepare the Data Frame to be readable
        """

        df_new = df.drop(['ID'], axis=0)
        df_new.columns = df_new.iloc[0,:]
        df_new.drop([np.nan], axis=0, inplace=True)
        df_new.columns.name = 'ID'

        return df_new


    # Function: Read clinical data
    def read_clinical_data(self, patient_data_od_path, patient_data_os_path):
        """
        Return excel data as pandas Data Frame
        """

        df_od = pd.read_excel(patient_data_od_path, index_col=[0])
        df_os = pd.read_excel(patient_data_os_path, index_col=[0])


        return self._fix_df(df=df_od), self._fix_df(df=df_os)



    # Function: Return three arrays of shape 488 with the diagnosis tag, eye ID (od, os) and patient ID
    def get_diagnosis(self, patient_data_od_path, patient_data_os_path):
        """
        Return three arrays of shape 488 with the diagnosis tag, eye ID (od, os)
        and patient ID
        """

        df_od, df_os = self.read_clinical_data(patient_data_od_path, patient_data_os_path)

        index_od = np.ones(df_od.iloc[:,2].values.shape, dtype=np.int8)
        index_os = np.zeros(df_os.iloc[:,2].values.shape, dtype=np.int8)

        eyeID = np.array(list(zip(index_od, index_os))).reshape(-1)
        tag = np.array(list(zip(df_od.iloc[:,2].values, df_os.iloc[:,2].values))).reshape(-1)
        patID = np.array([[int(i.replace('#', ''))] * 2 for i in df_od.index]).reshape(-1)


        return tag, eyeID, patID
