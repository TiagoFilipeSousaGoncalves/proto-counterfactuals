# Imports
import argparse
import os
import tqdm
import numpy as np
import pandas as pd
from PIL import Image



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="cub2002011-dataset", help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    parser.add_argument('--n_folds', type=int, default=5, help="Set the number of folds for cross-validation.")
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
            bbox_coord = bbox[1::]

            if img_id == bbox_id:
                image_ids.append(img_id)
                image_fnames.append(img_fname)
                bbox_coords.append(bbox_coord)
    # print(image_ids)
    # print(image_fnames)
    # print(bbox_coords)



    # Go through each fold
    for fold in range(args.n_folds):

        # Open data splits
        train_split = pd.read_csv(os.path.join(args.data_dir, "processed", f"kf_{fold}", "train.csv"))
        train_images_fnames = train_split.values

        val_split = pd.read_csv(os.path.join(args.data_dir, "processed", f"kf_{fold}", "val.csv"))
        val_images_fnames = val_split.values

        test_split = pd.read_csv(os.path.join(args.data_dir, "processed", f"kf_{fold}", "test.csv"))
        test_images_fnames = test_split.values


        # Go through images, train_test_split and bounding boxes
        for img_fname, img_bbox in tqdm.tqdm(zip(image_fnames, bbox_coords)):

            # Check if image is for train or test
            if img_fname in train_images_fnames:
                split = "train"
            elif img_fname in val_images_fnames:
                split = "val"
            else:
                split = "test"
            # print(img_fname)
            # print(img_bbox)
            # print(split)

            # Get bounding boxes info
            x = float(img_bbox[0])
            y = float(img_bbox[1])
            width = float(img_bbox[2])
            height = float(img_bbox[3])


            # Open image
            image_path = os.path.join(args.data_dir, "processed", f"kf_{fold}", split, "images", img_fname)
            pil_img = Image.open(image_path)
            # print(image_path)

            # Crop image
            crop_img = pil_img.crop((x, y, x+width, y+height))

            # Save image
            img_class_folder = img_fname.split("/")[0]
            os.makedirs(os.path.join(args.data_dir, "processed", f"kf_{fold}", split, "cropped", img_class_folder), exist_ok=True)
            crop_img.save(os.path.join(args.data_dir, "processed", f"kf_{fold}", split, "cropped", img_fname))