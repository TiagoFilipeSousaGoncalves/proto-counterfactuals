# Imports
import os
import shutil
import numpy as np

# Project Imports
from utilities import get_diagnosis

# Sklearn Imports
from sklearn.model_selection import train_test_split



# Read diagnostic labels
labels, eyeID, patID = get_diagnosis(
    patient_data_od_path=os.path.join("data", "papila", "raw", "ClinicalData", "patient_data_od.xlsx"),
    patient_data_os_path=os.path.join("data", "papila", "raw", "ClinicalData", "patient_data_os.xlsx")
)


# Some debug prints
# print(y)
# print(patID)
# print(eyeID)


# Get the unique patient indices
unique, unique_indices, unique_inverse, unique_counts = np.unique(ar=patID,  return_index=True, return_inverse=True, return_counts=True)
# print(unique_indices)


# Get X and y
X = patID[unique_indices]
y = labels[unique_indices]
# print(X, y)


# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# print(X_train, y_train)
# print(X_test, y_test)


# Create a directory for each data split and split images among folders
for split, patient_ids in zip(["train", "test"], [X_train, X_test]):
    
    # Get the split path
    split_dir = os.path.join("data", "papila", "processed", "splits", split)


    # Create directory
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    

    # Go through each patient ID
    for p_id in patient_ids:

        # Create folder images in split directory
        right_img = "RET%03dOD.png" % p_id
        left_img = "RET%03dOS.png" % p_id


        # Go through each image
        for img in [right_img, left_img]:
            
            # Copy folder to the split directory
            source = os.path.join("data", "papila", "processed", "rois", img)
            destination = os.path.join("data", "papila", "processed", "splits", split, img.split('.')[0], img)
            if not os.path.isdir(os.path.join("data", "papila", "processed", "splits", split, img.split('.')[0])):
                os.makedirs(os.path.join("data", "papila", "processed", "splits", split, img.split('.')[0]))

            _ = shutil.copyfile(source, destination)



print("Finished.")
