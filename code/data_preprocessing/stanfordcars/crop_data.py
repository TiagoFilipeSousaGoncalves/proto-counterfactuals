# Imports
import os
import scipy.io as sio
from PIL import Image



# Directories
data = "data"
stanfordcars = "stanfordcars"
car_devkit = "car_devkit"
cars_test = "cars_test"
cars_train = "cars_train"



# Process stuff
for cars_subset in [cars_train, cars_test]:

    # Get the directory of images
    images = os.path.join(data, stanfordcars, cars_subset, "images")

    # Create a directory for cropped images
    if not os.path.isdir(os.path.join(data, stanfordcars, cars_subset, "images_cropped")):
        os.makedirs(os.path.join(data, stanfordcars, cars_subset, "images_cropped"))


    # Get the correspondent .MAT file
    if cars_subset == cars_train:
        mat_file = sio.loadmat(os.path.join(data, stanfordcars, car_devkit, "devkit", f"{cars_subset}_annos.mat"))
    else:
        mat_file = sio.loadmat(os.path.join(data, stanfordcars, car_devkit, "devkit", f"{cars_subset}_annos_withlabels.mat"))
    # print(mat_file['annotations'])


    # Go through data points
    for entries in mat_file['annotations']:
        for sample in entries:
            # dtype=[('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('fname', 'O')])}
            bbox_x1 = sample[0][0,0]
            bbox_y1 = sample[1][0,0]
            bbox_x2 = sample[2][0,0]
            bbox_y2 = sample[3][0,0]
            label = sample[4][0,0]
            fname = sample[5][0]
            # print(sample)
            # print(bbox_x1, bbox_y1, bbox_x2, bbox_y2, label, fname)


            # Open the file
            pil_image = Image.open(os.path.join(data, stanfordcars, cars_subset, "images", fname)).convert('RGB')
            # pil_image.show()

            # Crop image
            pil_image_crop = pil_image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
            # pil_image_crop.show()

            # Save cropped image
            pil_image_crop.save(os.path.join(data, stanfordcars, cars_subset, "images_cropped", fname))



print("Finished")
