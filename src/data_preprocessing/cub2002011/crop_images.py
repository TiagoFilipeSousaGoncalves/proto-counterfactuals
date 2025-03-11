# Imports
import argparse
import os
import tqdm
import numpy as np 
from PIL import Image



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="cub2002011-dataset", help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    args = parser.parse_args()

    # Open images.txt
    images = np.genfromtxt(os.path.join(args.data_dir, "CUB_200_2011", "images.txt"), dtype=str)
    # print(images)

    # Open bounding_boxes.txt
    bounding_boxes = np.genfromtxt(os.path.join(args.data_dir, "CUB_200_2011", "bounding_boxes.txt"), dtype=str)
    # print(float(bounding_boxes[0,1]))

    # Create a list with all info
    image_ids, image_fnames, bbox_coords = list(), list(), list()
    for img in images:
        for bbox in bounding_boxes:
            img_id = img[0]
            img_fname = img[1]
            bbox_id = bbox[0]
            bbox_coord = bbox[1:4]

            if img_id == bbox_id:
                image_ids.append(img_id)
                image_fnames.append(img_fname)
                bbox_coords.append(bbox_coord)
    print(image_ids)
    print(image_fnames)
    print(bbox_coords)




    """
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

            crop_img.save(os.path.join(data, cub_200_2011, processed_data, split, cropped, image_info[1]))"
    """