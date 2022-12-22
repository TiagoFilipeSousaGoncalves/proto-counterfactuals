# Imports
import os
import tqdm
import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Project Imports
from utilities import contour_to_mask, apply_mask


# Directories and Files
if not os.path.isdir(os.path.join("data", "papila", "processed")):
    os.makedirs(os.path.join("data", "papila", "processed"))



# Get images
image_fnames = [i for i in os.listdir(os.path.join("data", "papila", "raw", "FundusImages")) if not i.startswith('.')]
# print(images)



# Go through images' fnames
for _, image_name in tqdm.tqdm(enumerate(image_fnames)):


    # Open image
    pil_img = Image.open(os.path.join("data", "papila", "raw", "FundusImages", image_name))
    npy_img = np.array(pil_img)


    # Get ROIs (We use both expert annotations)
    cup_path1 = os.path.join("data", "papila", "raw", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_cup_exp1.txt")
    disc_path1 = os.path.join("data", "papila", "raw", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_disc_exp1.txt")
    cup_path2 = os.path.join("data", "papila", "raw", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_cup_exp2.txt")
    disc_path2 = os.path.join("data", "papila", "raw", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_disc_exp2.txt")


    # Generate ROIs
    cup_mask1 = contour_to_mask(cup_path1, npy_img.shape)
    disc_mask1 = contour_to_mask(disc_path1, npy_img.shape)
    cup_mask2 = contour_to_mask(cup_path2, npy_img.shape)
    disc_mask2 = contour_to_mask(disc_path2, npy_img.shape)

    # Generate final mask
    final_mask = np.zeros_like(cup_mask1)
    final_mask[cup_mask1==1] = 1
    final_mask[disc_mask1==1] = 1
    final_mask[cup_mask2==1] = 1
    final_mask[disc_mask2==1] = 1
    # final_mask *= 255

    # Some debug operations
    # image = apply_mask(image=npy_img, mask=final_mask, color=(1.0, 1.0, 1.0))
    # plt.imshow(image, cmap="gray")
    # plt.show()


    # We have to be sure we are working with binary image
    ret, binary = cv2.threshold(final_mask*255, 127, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary, cmap="gray")
    # plt.show()

    # Get countours of this image
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
    # Extract bounding box from mask
    x, y, w, h = cv2.boundingRect(contours[0])


    # Crop image
    crop_img = pil_img.crop((x, y, x+w, y+h))
    # plt.imshow(np.array(crop_img), cmap="gray")
    # plt.show()


    # Some debug tests using OpenCV
    # crop_img = npy_img[y:y+h, x:x+w]
    # plt.imshow(np.array(crop_img), cmap="gray")
    # plt.show()

    # Save image(s)
    if not os.path.isdir(os.path.join("data", "papila", "processed", "rois")):
        os.makedirs(os.path.join("data", "papila", "processed", "rois"))


    crop_img.save(os.path.join("data", "papila", "processed", "rois", image_name.split('.')[0] + ".png"))



print("Finished")
