# Imports
import argparse
import os
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

    # Get the unique patient indices
    unique, unique_indices, unique_inverse, unique_counts = np.unique(ar=patID,  return_index=True, return_inverse=True, return_counts=True)

    # Get X and y
    X = patID[unique_indices]
    y = labels[unique_indices]

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
            'patient_ids': [X[i] for i in train_index],
            'images_labels': [y[i] for i in train_index]
        }
        test_dict[fold] = {
            'patient_ids': [X[i] for i in test_index],
            'images_labels': [y[i] for i in test_index]
        }
    for fold in range(args.n_folds):
        X_ = trainval_dict[fold]['patient_ids']
        y_ = trainval_dict[fold]['images_labels']
        for _, (train_index, test_index) in enumerate(sss_train_val.split(X_, y_)):
            train_dict[fold] = {
                'patient_ids': [X_[i] for i in train_index],
                'images_labels': [y_[i] for i in train_index]
            }
            val_dict[fold] = {
                'patient_ids': [X_[i] for i in test_index],
                'images_labels': [y_[i] for i in test_index]
            }


    # Create a CSV file with the data splits and save it into the processed folder
    data_splits = {
        'patient_ids':list(),
        'images_fnames':list(),
        'images_labels':list(),
        'split':list(),
        'fold':list()
    }

    # Go through each fold
    for fold in range(args.n_folds):
        assert len(X) == len(train_dict[fold]['patient_ids']) + len(val_dict[fold]['patient_ids']) + len(test_dict[fold]['patient_ids'])

        # Create fold directory
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)

        # Go through the number of image anatomic locations (2 per patient)
        for anatomic_location in ['OD', 'OS']:

            # Populate data_splits dictionary
            # Train
            data_splits['patient_ids'].extend(train_dict[fold]['patient_ids'])
            data_splits['images_fnames'].extend(["RET%03d%s.png" % (p_id, anatomic_location) for p_id in train_dict[fold]['patient_ids']])
            data_splits['images_labels'].extend(train_dict[fold]['images_labels'])
            data_splits['split'].extend(['train'] * len(train_dict[fold]['patient_ids']))
            data_splits['fold'].extend([fold] * len(train_dict[fold]['patient_ids']))

            # Val
            data_splits['patient_ids'].extend(val_dict[fold]['patient_ids'])
            data_splits['images_fnames'].extend(["RET%03d%s.png" % (p_id, anatomic_location) for p_id in val_dict[fold]['patient_ids']])
            data_splits['images_labels'].extend(val_dict[fold]['images_labels'])
            data_splits['split'].extend(['val'] * len(val_dict[fold]['patient_ids']))
            data_splits['fold'].extend([fold] * len(val_dict[fold]['patient_ids']))

            # Test
            data_splits['patient_ids'].extend(test_dict[fold]['patient_ids'])
            data_splits['images_fnames'].extend(["RET%03d%s.png" % (p_id, anatomic_location) for p_id in test_dict[fold]['patient_ids']])
            data_splits['images_labels'].extend(test_dict[fold]['images_labels'])
            data_splits['split'].extend(['test'] * len(test_dict[fold]['patient_ids']))
            data_splits['fold'].extend([fold] * len(test_dict[fold]['patient_ids']))



    # Save a dataframe with the splits
    splits_df = pd.DataFrame.from_dict(data_splits)
    splits_df.to_csv(os.path.join(data_dir, "processed", "data_splits.csv"), index=False)
