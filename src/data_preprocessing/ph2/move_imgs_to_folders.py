# Imports
import os
import shutil



# Directories
data = "data"
ph2 = "ph2"
processed_images = "processed_images"
train = "train"
test = "test"
cropped = "cropped"



# Process stuff
for subset in [train, test]:

    # Get the directory of images
    images_fpath = os.path.join(data, ph2, processed_images, subset, cropped)

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



print("Finished")
