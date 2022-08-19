# Imports
import os
import tqdm
import numpy as np 
from PIL import Image



# Directories and Files
data = "data"
cub_200_2011 = "cub_200_2011"
source_data = "source_data"
images_txt = "images.txt"
train_test_split_txt = "train_test_split.txt"
bounding_boxes_txt = "bounding_boxes.txt"


# Make new directories and files
processed_data = "processed_data"
train = "train"
test = "test"
cropped = "cropped"



# Open images.txt
images = np.genfromtxt(os.path.join(data, cub_200_2011, source_data, images_txt), dtype=str)
# print(images)

# Open train_test_split.txt
train_test_split = np.genfromtxt(os.path.join(data, cub_200_2011, source_data, train_test_split_txt), dtype=str)
# print(train_test_split)

# Open bounding_boxes.txt
bounding_boxes = np.genfromtxt(os.path.join(data, cub_200_2011, source_data, bounding_boxes_txt), dtype=str)
# print(float(bounding_boxes[0,1]))



# Go through images, train_test_split and bounding boxes
for image_info, train_test_info, bounding_box_info in tqdm.tqdm(zip(images, train_test_split, bounding_boxes)):

    # Assure Image IDs are the same
    if (image_info[0] == train_test_info[0]) and (image_info[0] == bounding_box_info[0]):

        # Check if image is for train or test
        split = "train" if train_test_info[1] == '1' else "test"


        # Get bounding boxes info
        x = float(bounding_box_info[1])
        y = float(bounding_box_info[2])
        width = float(bounding_box_info[3])
        height = float(bounding_box_info[4])


        # Open image
        image_path = os.path.join(data, cub_200_2011, processed_data, split, "images", image_info[1])
        pil_img = Image.open(image_path)

        # Crop image
        crop_img = pil_img.crop((x, y, x+width, y+height))

        # Save image
        img_class_folder = image_info[1].split("/")[0]
        if not os.path.isdir(os.path.join(data, cub_200_2011, processed_data, split, cropped, img_class_folder)):
            os.makedirs(os.path.join(data, cub_200_2011, processed_data, split, cropped, img_class_folder))

        crop_img.save(os.path.join(data, cub_200_2011, processed_data, split, cropped, image_info[1]))


print("Finished")
