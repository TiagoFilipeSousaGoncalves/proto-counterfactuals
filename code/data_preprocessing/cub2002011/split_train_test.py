# Imports
import os
import shutil
import tqdm
import numpy as np 



# Directories and Files
data = "data"
cub_200_2011 = "cub2002011"
source_data = "source_data"
images_txt = "images.txt"
train_test_split_txt = "train_test_split.txt"


# Make new directories and files
processed_data = "processed_data"
train = "train"
test = "test"
cropped = "cropped"

# Create new directories
for directory in [train, test]:
    if not os.path.isdir(os.path.join(data, cub_200_2011, processed_data, directory, cropped)):
        os.makedirs(os.path.join(data, cub_200_2011, processed_data, directory, cropped))



# Open images.txt
images = np.genfromtxt(os.path.join(data, cub_200_2011, source_data, images_txt), dtype=str)
# print(images)

# Open train_test_split.txt
train_test_split = np.genfromtxt(os.path.join(data, cub_200_2011, source_data, train_test_split_txt), dtype=str)
# print(train_test_split)


# Let's split the data
# print(next(zip(images, train_test_split)))
for image_info, train_test_info in tqdm.tqdm(zip(images, train_test_split)):
    # print(f"Image ID: {image_info[0]}")
    # print(f"Image Path: {image_info[1]}")
    # print(f"Image: ID: {train_test_info[0]}")
    # print(f"Train (1) or Test (0): {train_test_info[1]}")


    # Assure Image IDs are the same
    if image_info[0] == train_test_info[0]:

        # Check if image is for train or test
        split = "train" if train_test_info[1] == '1' else "test"
        

        # Create a folder for the complete images
        if not os.path.isdir(os.path.join(data, cub_200_2011, processed_data, split, "images")):
            os.makedirs(os.path.join(data, cub_200_2011, processed_data, split, "images"))
        

        # Get image class folder
        img_class_folder = image_info[1].split("/")[0]

        # Create this folder (if it does not exist)
        if not os.path.isdir(os.path.join(data, cub_200_2011, processed_data, split, "images", img_class_folder)):
            os.makedirs(os.path.join(data, cub_200_2011, processed_data, split, "images", img_class_folder))
        
        # Copy this image to this split folder
        src = os.path.join(data, cub_200_2011, source_data, "images", image_info[1])
        dst = os.path.join(data, cub_200_2011, processed_data, split, "images", image_info[1])
        shutil.copy(src, dst)


print("Finished")
