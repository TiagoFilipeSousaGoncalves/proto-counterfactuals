# Imports
import os
import shutil
import tqdm
import numpy as np
import pandas as pd

# Sklearn Imports
from sklearn.model_selection import train_test_split



# Random seeds
random_seed = 42



# Directories and filenames
data_dir = "data"
ph2_dir = os.path.join(data_dir, "ph2")
ph2_xlsx = "PH2_dataset.xlsx"


# Open PH2 XLSX file
ph2_df = pd.read_excel(os.path.join(ph2_dir, ph2_xlsx), skiprows=[i for i in range(12)])
# print(ph2_df.head())
# print(ph2_df.columns)

# Get only classification columns
ph2_df = ph2_df.copy()[['Image Name', 'Common Nevus', 'Atypical Nevus', 'Melanoma']]
# print(ph2_df.head())


# Add an extra column
diagnosis_dict = {
    "Common Nevus":0,
    "Atypical Nevus":1,
    "Melanoma":2
}

# Add column "Label"
ph2_df["Label"] = -1
# print(ph2_df.head())


# Go through the DataFrame
ph2_df = ph2_df.copy().reset_index()

for index, row in ph2_df.iterrows():
    
    # Get values
    if row['Common Nevus'] == "X":
        ph2_df.iloc[index, -1] = diagnosis_dict['Common Nevus']

    elif row['Atypical Nevus'] == "X":
        ph2_df.iloc[index, -1] = diagnosis_dict['Atypical Nevus']
    
    elif row['Melanoma'] == "X":
        ph2_df.iloc[index, -1] = diagnosis_dict['Melanoma']
    

    # print(ph2_df.iloc[index])



# Get X, y
X, y = ph2_df.copy()['Image Name'].values, ph2_df.copy()['Label'].values
# print(X, y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_seed)
# print(X_train, y_train)
# print(X_test, y_test)



# Create a directory for processed data
processed_images_dir = os.path.join(ph2_dir, "processed_images")
if not os.path.isdir(processed_images_dir):
    os.makedirs(processed_images_dir)


# Create a directory for each data split and split images among folders
for split, image_names in zip(["train", "test"], [X_train, X_test]):
    
    # Get the split path
    split_dir = os.path.join(processed_images_dir, split, "raw")


    # Create directory
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    # print(split, image_names)

    # Go through the image names
    for img_name in image_names:

        # Copy folder to the split directory
        source = os.path.join(ph2_dir, "images", img_name)
        destination = os.path.join(split_dir, img_name)

        _ = shutil.copytree(source, destination)



print("Finished.")
