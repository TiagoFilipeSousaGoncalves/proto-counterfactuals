# Imports
import argparse
import os
import shutil
import pandas as pd

# Sklearn Imports
from sklearn.model_selection import train_test_split



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    parser.add_argument('--train_size', type=float, default=0.70, help="Set the train size (%).")
    parser.add_argument('--val_size', type=float, default=0.10, help="Set the validation size (%)")
    parser.add_argument('--test_size', type=float, default=0.20, help="Set the test size (%).")
    args = parser.parse_args()

    data_dir = args.data_dir

    # Open PH2 XLSX file
    ph2_df = pd.read_excel(os.path.join(data_dir, "metadata", "PH2_dataset.xlsx"), skiprows=[i for i in range(12)])
    # print(ph2_df.head())
    # print(ph2_df.columns)

    # Get only classification columns
    ph2_df = ph2_df.copy()[['Image Name', 'Common Nevus', 'Atypical Nevus', 'Melanoma']]
    # print(ph2_df.head())


    # Add an extra column
    diagnosis_dict = {
        "Common Nevus":0,
        "Atypical Nevus":1,
        "Melanoma":2
    }

    # Add column "Label"
    ph2_df["Label"] = -1
    # print(ph2_df.head())


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
        

        # print(ph2_df.iloc[index])



    # Create a directory for processed data
    processed_images_dir = os.path.join(args.data_dir, "processed")
    if not os.path.isdir(processed_images_dir):
        os.makedirs(processed_images_dir)



    # Get X, y
    X, y = ph2_df.copy()['Image Name'].values, ph2_df.copy()['Label'].values
    # print(X, y)

    # Split into train and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    assert (args.train_size + args.val_size + args.test_size) == 1.0
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=(args.train_size + args.val_size), random_state=args.seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=(args.train_size/(args.train_size + args.val_size)), random_state=args.seed, stratify=y_train_val)
    assert round(len(X_train)/len(X), 2) == args.train_size
    assert round(len(X_val)/len(X), 2) == args.val_size
    assert round(len(X_test)/len(X), 2) == args.test_size
    # print(X_train, y_train)
    # print(X_test, y_test)

    # Save this into .CSVs
    X_train_df = pd.DataFrame.from_dict({'train':list(X_train)})
    X_train_df.to_csv(os.path.join(data_dir, "processed", "train.csv"), index=False)

    X_val_df = pd.DataFrame.from_dict({'val':list(X_val)})
    X_val_df.to_csv(os.path.join(data_dir, "processed", "val.csv"), index=False)

    X_test_df = pd.DataFrame.from_dict({'test':list(X_test)})
    X_test_df.to_csv(os.path.join(data_dir, "processed", "test.csv"), index=False)


    # Create a directory for processed data
    processed_images_dir = os.path.join(args.data_dir, "processed", "images")
    if not os.path.isdir(processed_images_dir):
        os.makedirs(processed_images_dir)


    # Create a directory for each data split and split images among folders
    for split, image_names in zip(["train", "val", "test"], [X_train, X_val, X_test]):
        
        # Get the split path
        split_dir = os.path.join(processed_images_dir, split, "raw")


        # Create directory
        if not os.path.isdir(split_dir):
            os.makedirs(split_dir)

        # print(split, image_names)

        # Go through the image names
        for img_name in image_names:

            # Copy folder to the split directory
            source = os.path.join(args.data_dir, "images", img_name)
            destination = os.path.join(split_dir, img_name)
            _ = shutil.copytree(source, destination)