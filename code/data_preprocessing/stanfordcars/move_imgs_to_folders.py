# Imports
import os
import shutil



# Directories
data = "data"
stanfordcars = "stanfordcars"
car_devkit = "car_devkit"
cars_test = "cars_test"
cars_train = "cars_train"



# Process stuff
for cars_subset in [cars_train, cars_test]:

    # Get the directory of images
    images_fpath = os.path.join(data, stanfordcars, cars_subset, "images_cropped")

    # Get images' fnames
    images_fnames = os.listdir(images_fpath)

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



print("Finished")
