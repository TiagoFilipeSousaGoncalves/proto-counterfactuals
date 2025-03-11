# Imports
import argparse
import os
import shutil
import tqdm
import numpy as np

# Sklearn Imports
from sklearn.model_selection import train_test_split



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="cub2002011-dataset", help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    parser.add_argument('--train_size', type=float, default=0.70, help="Set the train size (%).")
    parser.add_argument('--val_size', type=float, default=0.10, help="Set the validation size (%)")
    parser.add_argument('--test_size', type=float, default=0.20, help="Set the test size (%).")
    args = parser.parse_args()


    # Directories and Files
    data_dir = args.data_dir

    # Create new directories
    for directory in ["train", "val", "test"]:
        if not os.path.isdir(os.path.join(data_dir, "processed", directory, "cropped")):
            os.makedirs(os.path.join(data_dir, "processed", directory, "cropped"))



    # Open images.txt
    images = np.genfromtxt(os.path.join(data_dir, "CUB_200_2011", "images.txt"), dtype=str)
    # print(images)

    # Note: We will build a train-test split with validation, so, we will do it from scratch
    # Open train_test_split.txt
    # train_test_split = np.genfromtxt(os.path.join(data_dir, "CUB_200_2011", "train_test_split.txt"), dtype=str)
    # print(train_test_split)

    # Create a new list with the ID's and classes
    images_ids = list()
    image_fnames = list()
    images_classes = list()
    for p in images:
        img_id = p[0]
        img_fname = p[1]
        img_cl = img_fname.split('.')[0]
        images_ids.append(img_id)
        image_fnames.append(img_fname)
        images_classes.append(img_cl)
    # print(images_ids)
    # print(images_classes)
    # print(len(images_ids), len(images_classes), np.unique(images_classes))

    assert (args.train_size + args.val_size + args.test_size) == 1.0
    X_train_val, X_test, y_train_val, y_test = train_test_split(image_fnames, images_classes, train_size=(args.train_size + args.val_size), random_state=args.seed, stratify=images_classes)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=(args.train_size/(args.train_size + args.val_size)), random_state=args.seed, stratify=y_train_val)
    print(len(X_train)/len(image_fnames), len(X_val)/len(image_fnames), len(X_test)/len(image_fnames))

    """
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
        """