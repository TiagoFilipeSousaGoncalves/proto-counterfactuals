# Imports
import argparse
import os
import shutil



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    args = parser.parse_args()

    # Directories
    data_dir = args.data_dir


    # Process stuff
    for subset in ["train", "val", "test"]:

        # Get the directory of images
        images_fpath = os.path.join(data_dir, "processed", "images", subset, "cropped")

        # Get images' fnames
        images_fnames = [i for i in os.listdir(images_fpath) if not i.startswith('.')]

        # Create folders for each images
        for fname in images_fnames:

            # Get folder's name
            folder_name = fname.split('.')[0]

            # Create folder
            folder_path = os.path.join(images_fpath, folder_name)
            os.mkdir(folder_path)

            # Move image to the folder
            src_path = os.path.join(images_fpath, fname)
            dst_path = os.path.join(folder_path, fname)
            shutil.move(src_path, dst_path)