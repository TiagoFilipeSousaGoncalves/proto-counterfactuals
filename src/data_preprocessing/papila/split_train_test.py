# Imports
import argparse
import os
import shutil
import numpy as np
import pandas as pd

# Project Imports
from utilities import get_diagnosis

# Sklearn Imports
from sklearn.model_selection import StratifiedShuffleSplit



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    parser.add_argument('--train_size', type=float, default=0.70, help="Set the train size (%).")
    parser.add_argument('--val_size', type=float, default=0.10, help="Set the validation size (%)")
    parser.add_argument('--test_size', type=float, default=0.20, help="Set the test size (%).")
    parser.add_argument('--n_folds', type=int, default=5, help="Set the number of folds for cross-validation.")
    args = parser.parse_args()


    # Directories and Files
    data_dir = args.data_dir


    # Read diagnostic labels
    labels, eyeID, patID = get_diagnosis(
        patient_data_od_path=os.path.join(data_dir, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ClinicalData", "patient_data_od.xlsx"),
        patient_data_os_path=os.path.join(data_dir, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ClinicalData", "patient_data_os.xlsx")
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


    # TODO: Erase after testing
    # Split into train and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    # assert (args.train_size + args.val_size + args.test_size) == 1.0
    # X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=(args.train_size + args.val_size), random_state=args.seed, stratify=y)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=(args.train_size/(args.train_size + args.val_size)), random_state=args.seed, stratify=y_train_val)
    # assert round(len(X_train)/len(X), 2) == args.train_size
    # assert round(len(X_val)/len(X), 2) == args.val_size
    # assert round(len(X_test)/len(X), 2) == args.test_size
    # print(X_train, y_train)
    # print(X_test, y_test)

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



    # Go through each fold
    for fold in range(args.n_folds):
        assert len(X) == len(train_dict[fold]['image_fnames']) + len(val_dict[fold]['image_fnames']) + len(test_dict[fold]['image_fnames'])

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


        # Create a directory for each data split and split images among folders
        for split, patient_ids in zip(["train", "val", "test"], [X_train, X_val, X_test]):
            
            # Get the split path
            split_dir = os.path.join(data_dir, "processed", f"kf_{fold}", "splits", split)


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
                    source = os.path.join(data_dir, "processed", "rois", img)
                    destination = os.path.join(data_dir, "processed", f"kf_{fold}", "splits", split, img.split('.')[0], img)
                    if not os.path.isdir(os.path.join(data_dir, "processed", f"kf_{fold}", "splits", split, img.split('.')[0])):
                        os.makedirs(os.path.join(data_dir, "processed", f"kf_{fold}", "splits", split, img.split('.')[0]))

                    _ = shutil.copyfile(source, destination)