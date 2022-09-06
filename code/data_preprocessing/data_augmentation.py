# Source: https://github.com/fanconic/this-does-not-look-like-that/blob/master/src/data/img_aug.py



# Imports
import os
import Augmentor



# Function: Perform 40x data augmentation for each training image.
def augment(source_dir, target_dir):


    # source_dir = datasets_root_dir + "train_cropped/"
    # target_dir = datasets_root_dir + "train_cropped_augmented/"


    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)


    folders = [os.path.join(source_dir, folder) for folder in next(os.walk(source_dir))[1]]
    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(source_dir))[1]]

    print(target_folders)

    exit()


    # Iterate through all the folders
    for i in range(len(folders)):
        fd = folders[i]
        tfd = target_folders[i]

        # Rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        # Skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        # Shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p

        # Random_distortion
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p



# Run this file to proceed with the data augmentation
if __name__ == "__main__":

    # CUB2002011
    CUB_SRC_DIR = "data/cub2002011/processed_data/train/cropped/"
    CUB_TARGET_DIR = "data/cub2002011/processed_data/train/cropped_augmented/"
    augment(source_dir=CUB_SRC_DIR, target_dir=CUB_TARGET_DIR)

    print("Finished.")
