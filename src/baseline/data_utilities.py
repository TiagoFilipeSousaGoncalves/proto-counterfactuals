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



# Dataset
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
    def __init__(self, data_path, split="train", augmented=True, transform=None):

        """
        Args:
            data_path (string): Data directory.
            classes_txt (string): File with the classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        assert split in ("train", "val", "test")
        self.split = split
        split_path = os.path.join(data_path, "processed", split, "cropped")

        if augmented:
            assert split == "train"

        # Data path (get images folders)
        images_folders = [f for f in os.listdir(split_path) if not f.startswith('.')]
        
        

        # Enter each folder and add the image path to our images_fpaths variable
        images_fpaths = list()
        for folder in images_folders:
            
            img_fnames_aug = list()
            img_fnames = list()

            # Get images
            if augmented:
                img_fnames_aug += [i for i in os.listdir(os.path.join(split_path, folder, "augmented")) if not i.startswith('.')]
            img_fnames += [i for i in os.listdir(os.path.join(split_path, folder)) if not i.startswith('.')]


            # Get each image
            for img_name in img_fnames:                    
                images_fpaths.append(os.path.join(folder, img_name))
            if augmented:
                for img_name in img_fnames_aug:
                    images_fpaths.append(os.path.join(folder, "augmented", img_name))


        # Clean images_fpaths (to prevent IsADirectoryError errors)
        images_fpaths = [path for path in images_fpaths if not os.path.isdir(os.path.join(split_path, path))]
        

        # Add this to our variables
        self.data_path = data_path
        self.images_fpaths = images_fpaths


        # Extract labels from data path
        labels = np.genfromtxt(os.path.join(data_path, "CUB_200_2011", "classes.txt"), dtype=str)
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
        image = Image.open(os.path.join(self.data_path, "processed", self.split, "cropped", img_path)).convert('RGB')

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



# PAPILADataset: Dataset Class
class PAPILADataset(Dataset):
    def __init__(self, data_path, subset, cropped, augmented, transform=None):

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
            self.images_dir = os.path.join(data_path, "processed", "splits", subset)
        
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
