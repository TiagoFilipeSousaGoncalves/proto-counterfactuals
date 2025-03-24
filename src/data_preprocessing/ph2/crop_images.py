# Imports
import argparse
import os
import numpy as np 
from PIL import Image
import cv2



if __name__ == "__main__":

    # Command Line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    args = parser.parse_args()

    data_dir = args.data_dir
    ph2_proc_dir = os.path.join(data_dir, "processed", "images")


    # Go through several splits and directories
    for split in ["train", "val", "test"]:

        # Create "cropped" directory
        cropped_dir = os.path.join(ph2_proc_dir, split, "cropped")
        if not os.path.isdir(cropped_dir):
            os.makedirs(cropped_dir)
        

        # Go through "raw" directory
        raw_dir = os.path.join(ph2_proc_dir, split, "raw")


        # Get directories of "raw" directory
        image_dirs = [d for d in os.listdir(raw_dir) if not d.startswith('.')]

        # Go through images
        for image_name in image_dirs:

            # Open original file
            pil_image = Image.open(os.path.join(raw_dir, image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp")).convert('RGB')

            # Open mask
            pil_mask = Image.open(os.path.join(raw_dir, image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp")).convert('L')

            
            # Convert PIL mask to NumPy array
            npy_mask = np.array(pil_mask.copy())

            # We have to be sure we are working with binary image
            ret, binary = cv2.threshold(npy_mask, 127, 255, cv2.THRESH_BINARY)

            # Get countours of this image
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract bounding box from mask
            x, y, w, h = cv2.boundingRect(contours[0])

            # Crop image
            crop_img = pil_image.copy().crop((x, y, x+w, y+h))


            # Save image
            crop_img.save(os.path.join(cropped_dir, f"{image_name}.png"))