# Source: https://github.com/fanconic/this-does-not-look-like-that/blob/master/src/data/img_aug.py



# Imports
import os
import argparse
import Augmentor



# Function: Perform 40x data augmentation for each training image.
def augment(source_dir):


    # source_dir = datasets_root_dir + "train_cropped/"
    # target_dir = datasets_root_dir + "train_cropped_augmented/"

    # Create new directories (if needed)
    # if not os.path.isdir(target_dir):
    #     os.makedirs(target_dir)


    # Get image names
    img_folders = [f for f in os.listdir(source_dir) if not f.startswith('.')]
    img_folders.sort()
    # source_folders = [os.path.join(source_dir, folder) for folder in next(os.walk(source_dir))[1]]
    
    # Get source folders
    source_folders = [os.path.join(source_dir, folder) for folder in img_folders]
    
    # Get target folders
    # target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(source_dir))[1]]
    # target_folders = [os.path.join(target_dir, folder) for folder in img_folders]


    # Debug prints
    # print(source_folders)
    # print(target_folders)


    # Iterate through all the folders
    for i in range(len(source_folders)):
        fd = source_folders[i]
        # tfd = target_folders[i]

        # Rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory="augmented")
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        # Skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory="augmented")
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        # Shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory="augmented")
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        # Random_distortion
        p = Augmentor.Pipeline(source_directory=fd, output_directory="augmented")
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p



# Run this file to proceed with the data augmentation
if __name__ == "__main__":

    # CLI Interface
    # Data set
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PH2, STANFORDCARS.")

    # Parse the arguments
    args = parser.parse_args()

    if args.dataset == "CUB2002011":
        # CUB2002011
        CUB_SRC_DIR = "data/cub2002011/processed_data/train/cropped"
        augment(source_dir=CUB_SRC_DIR)

    elif args.dataset == "STANFORDCARS":
        # STANFORDCARS
        STANFORDCARS_SRC_DIR = "data/stanfordcars/cars_train/images_cropped"
        augment(source_dir=STANFORDCARS_SRC_DIR)
    
    elif args.dataset == "PH2":
        # PH2
        PH2_SRC_DIR = "data/ph2/processed_images/train/cropped"
        augment(source_dir=PH2_SRC_DIR)
    
    else:
        pass



    print("Finished.")
