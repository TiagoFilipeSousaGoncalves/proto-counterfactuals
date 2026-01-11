# Imports
import argparse
import os
import pandas as pd

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



    # Load and process the PH2 metadata file
    ph2_df = pd.read_excel(os.path.join(args.data_dir, "metadata", "PH2_dataset.xlsx"), skiprows=[i for i in range(12)])
    ph2_df = ph2_df.copy()[['Image Name', 'Common Nevus', 'Atypical Nevus', 'Melanoma']]
    diagnosis_dict = {
        "Common Nevus":0,
        "Atypical Nevus":1,
        "Melanoma":2
    }

    # Add column "Label"
    ph2_df["Label"] = -1

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



    # Get X, y
    X, y = ph2_df.copy()['Image Name'].values, ph2_df.copy()['Label'].values

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
            'image_name': [X[i] for i in train_index],
            'image_label': [y[i] for i in train_index]
        }
        test_dict[fold] = {
            'image_name': [X[i] for i in test_index],
            'image_label': [y[i] for i in test_index]
        }
    for fold in range(args.n_folds):
        X_ = trainval_dict[fold]['image_name']
        y_ = trainval_dict[fold]['image_label']
        for _, (train_index, test_index) in enumerate(sss_train_val.split(X_, y_)):
            train_dict[fold] = {
                'image_name': [X_[i] for i in train_index],
                'image_label': [y_[i] for i in train_index]
            }
            val_dict[fold] = {
                'image_name': [X_[i] for i in test_index],
                'image_label': [y_[i] for i in test_index]
            }


    # Create a CSV file with the data splits and save it into the processed folder
    data_splits = {
        'image_name':list(),
        'image_label':list(),
        'split':list(),
        'fold':list()
    }

    # Go through each fold
    for fold in range(args.n_folds):
        assert len(X) == len(train_dict[fold]['image_name']) + len(val_dict[fold]['image_name']) + len(test_dict[fold]['image_name'])

        # Create fold directory
        os.makedirs(os.path.join(args.data_dir, "processed"), exist_ok=True)


        # Populate data_splits dictionary
        # Train
        data_splits['image_name'].extend(train_dict[fold]['image_name'])
        data_splits['image_label'].extend(train_dict[fold]['image_label'])
        data_splits['split'].extend(['train'] * len(train_dict[fold]['image_name']))
        data_splits['fold'].extend([fold] * len(train_dict[fold]['image_name']))

        # Val
        data_splits['image_name'].extend(val_dict[fold]['image_name'])
        data_splits['image_label'].extend(val_dict[fold]['image_label'])
        data_splits['split'].extend(['val'] * len(val_dict[fold]['image_name']))
        data_splits['fold'].extend([fold] * len(val_dict[fold]['image_name']))

        # Test
        data_splits['image_name'].extend(test_dict[fold]['image_name'])
        data_splits['image_label'].extend(test_dict[fold]['image_label'])
        data_splits['split'].extend(['test'] * len(test_dict[fold]['image_name']))
        data_splits['fold'].extend([fold] * len(test_dict[fold]['image_name']))


    # Save a dataframe with the splits
    splits_df = pd.DataFrame.from_dict(data_splits)
    splits_df.to_csv(os.path.join(args.data_dir, "processed", "data_splits.csv"), index=False)
