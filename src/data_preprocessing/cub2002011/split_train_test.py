# Imports
import argparse
import os
import shutil
import tqdm
import numpy as np
import pandas as pd

# Sklearn Imports
from sklearn.model_selection import StratifiedShuffleSplit



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="cub2002011-dataset", help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    parser.add_argument('--train_size', type=float, default=0.70, help="Set the train size (%).")
    parser.add_argument('--val_size', type=float, default=0.10, help="Set the validation size (%)")
    parser.add_argument('--test_size', type=float, default=0.20, help="Set the test size (%).")
    parser.add_argument('--n_folds', type=int, default=5, help="Set the number of folds for cross-validation.")
    args = parser.parse_args()


    # Directories and Files
    data_dir = args.data_dir

    # TODO: Remove this uppon testing
    # for directory in ["train", "val", "test"]:
    #     if not os.path.isdir(os.path.join(data_dir, "processed", directory, "cropped")):
    #         os.makedirs(os.path.join(data_dir, "processed", directory, "cropped"))



    # Open images.txt
    images = np.genfromtxt(os.path.join(data_dir, "CUB_200_2011", "images.txt"), dtype=str)
    # print(images)

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



    # Apply 5-fold cross validation split strategy
    assert (args.train_size + args.val_size + args.test_size) == 1.0

    # Create dictionaries to save the splits
    trainval_dict = dict()
    test_dict = dict()
    train_dict = dict()
    val_dict = dict()

    sss_trainval_test = StratifiedShuffleSplit(
        n_splits=args.n_folds,
        train_size=args.train_size + args.val_size,
        random_state=args.seed
    )
    sss_train_val = StratifiedShuffleSplit(
        n_splits=1,
        train_size=(args.train_size / (args.train_size + args.val_size)),
        random_state=args.seed
    )

    X = image_fnames
    y = images_classes
    for fold, (train_index, test_index) in enumerate(sss_trainval_test.split(X, y)):
        trainval_dict[fold] = {
            'image_fnames': [X[i] for i in train_index],
            'images_classes': [y[i] for i in train_index]
        }
        test_dict[fold] = {
            'image_fnames': [X[i] for i in test_index],
            'images_classes': [y[i] for i in test_index]
        }
    for fold in range(args.n_folds):
        X = trainval_dict[fold]['image_fnames']
        y = trainval_dict[fold]['images_classes']
        for _, (train_index, test_index) in enumerate(sss_train_val.split(X, y)):
            train_dict[fold] = {
                'image_fnames': [X[i] for i in train_index],
                'images_classes': [y[i] for i in train_index]
            }
            val_dict[fold] = {
                'image_fnames': [X[i] for i in test_index],
                'images_classes': [y[i] for i in test_index]
            }

    # TODO: Remove after testing
    # X_train_val, X_test, y_train_val, y_test = train_test_split(image_fnames, images_classes, train_size=(args.train_size + args.val_size), random_state=args.seed, stratify=images_classes)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=(args.train_size/(args.train_size + args.val_size)), random_state=args.seed, stratify=y_train_val)
    # assert round(len(X_train)/len(image_fnames), 2) == args.train_size
    # assert round(len(X_val)/len(image_fnames), 2) == args.val_size
    # assert round(len(X_test)/len(image_fnames), 2) == args.test_size


    # Go through each fold
    for fold in range(args.n_folds):
        assert len(image_fnames) == len(train_dict[fold]['image_fnames']) + len(val_dict[fold]['image_fnames']) + len(test_dict[fold]['image_fnames'])

        # Create fold directory
        os.makedirs(os.path.join(data_dir, "processed", f"kf_{fold}"), exist_ok=True)

        # Get the data splits
        X_train = train_dict[fold]['image_fnames']
        X_val = val_dict[fold]['image_fnames']
        X_test = test_dict[fold]['image_fnames']

        # Save this into .CSVs
        X_train_df = pd.DataFrame.from_dict({'train':list(X_train)})
        X_train_df.to_csv(os.path.join(data_dir, "processed", f"kf_{fold}", "train.csv"), index=False)

        X_val_df = pd.DataFrame.from_dict({'val':list(X_val)})
        X_val_df.to_csv(os.path.join(data_dir, "processed", f"kf_{fold}", "val.csv"), index=False)

        X_test_df = pd.DataFrame.from_dict({'test':list(X_test)})
        X_test_df.to_csv(os.path.join(data_dir, "processed", f"kf_{fold}", "test.csv"), index=False)


        # Let's split the data
        for split_idx, data_split in enumerate([X_train, X_val, X_test]):
            for image_fname in tqdm.tqdm(data_split):
                # print(image_fname)

                # Check if image is for train or test
                if split_idx == 0:
                    split_dir = "train"
                elif split_idx == 1:
                    split_dir = "val"
                else:
                    split_dir = "test"
                

                # Create a folder for the complete images
                os.makedirs(os.path.join(data_dir, "processed", f"kf_{fold}", split_dir, "images"), exist_ok=True)
                

                # Get image class folder
                img_class_folder = image_fname.split("/")[0]
                # print(img_class_folder)

                # Create this folder (if it does not exist)
                os.makedirs(os.path.join(data_dir, "processed", f"kf_{fold}", split_dir, "images", img_class_folder), exist_ok=True)

                
                # Copy this image to this split folder
                src = os.path.join(data_dir, "CUB_200_2011", "images", image_fname)
                dst = os.path.join(data_dir, "processed", f"kf_{fold}", split_dir, "images", image_fname)
                # print(src)
                # print(dst)
                shutil.copy(src, dst)