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

    # Open images.txt
    images = np.genfromtxt(os.path.join(data_dir, "CUB_200_2011", "images.txt"), dtype=str)


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


    # Create a CSV file with the data splits and save it into the processed folder
    data_splits = {
        'image_fnames':list(),
        'images_classes':list(),
        'split':list(),
        'fold':list()
    }

    # Go through each fold
    for fold in range(args.n_folds):
        assert len(image_fnames) == len(train_dict[fold]['image_fnames']) + len(val_dict[fold]['image_fnames']) + len(test_dict[fold]['image_fnames'])

        # Create fold directory
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)

        # Populate data_splits dictionary
        # Train
        data_splits['image_fnames'].extend(train_dict[fold]['image_fnames'])
        data_splits['images_classes'].extend(train_dict[fold]['images_classes'])
        data_splits['split'].extend(['train'] * len(train_dict[fold]['image_fnames']))
        data_splits['fold'].extend([fold] * len(train_dict[fold]['image_fnames']))

        # Val
        data_splits['image_fnames'].extend(val_dict[fold]['image_fnames'])
        data_splits['images_classes'].extend(val_dict[fold]['images_classes'])
        data_splits['split'].extend(['val'] * len(val_dict[fold]['image_fnames']))
        data_splits['fold'].extend([fold] * len(val_dict[fold]['image_fnames']))

        # Test
        data_splits['image_fnames'].extend(test_dict[fold]['image_fnames'])
        data_splits['images_classes'].extend(test_dict[fold]['images_classes'])
        data_splits['split'].extend(['test'] * len(test_dict[fold]['image_fnames']))
        data_splits['fold'].extend([fold] * len(test_dict[fold]['image_fnames']))



    # Save a dataframe with the splits
    splits_df = pd.DataFrame.from_dict(data_splits)
    splits_df.to_csv(os.path.join(data_dir, "processed", "data_splits.csv"), index=False)
