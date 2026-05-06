# Imports
import os
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset



# Function: Resize images
def resize_images(datapath, newpath, newheight=512):

    # Create new directories (if necessary)
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    # Go through data directory and generate new (resized) images
    for f in tqdm(os.listdir(datapath)):
        if(f.endswith(".jpg") or f.endswith('.png')):
            img = Image.open(os.path.join(datapath, f))
            w, h = img.size
            ratio = w / h
            new_w = int(np.ceil(newheight * ratio))
            new_img = img.resize((new_w, newheight), Image.ANTIALIAS)
            new_img.save(os.path.join(newpath, f))


    return



# Function: Apply normalisation to a given input function
def preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    assert x.size(1) == 3

    y = torch.zeros_like(x)

    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]

    return y



# Function: Allocate new tensor like x and apply the normalization used in the pretrained model
def preprocess_input_function(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    return preprocess(x=x, mean=mean, std=std)



# Function: Denormalise input(s), i.e., in this case images
def undo_preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    assert x.size(1) == 3

    y = torch.zeros_like(x)

    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]

    return y



# Function: Allocate new tensor like x and undo the normalization used in the pretrained model
def undo_preprocess_input_function(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    return undo_preprocess(x=x, mean=mean, std=std)



# Function: Save preprocessed image(s)
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    plt.imsave(fname, undo_preprocessed_img)

    return undo_preprocessed_img



# Function: Save image prototypes
def save_prototype(prototype_fname, prototypes_img_dir, index):

    try:
        # Note: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
        # Open an image file and save it
        # p_img = plt.imread(os.path.join(load_img_dir, 'prototype-img'+str(index)+'.png'))
        p_img = Image.open(os.path.join(prototypes_img_dir, 'prototype-img'+str(index)+'.png')).convert("RGB")
        p_img = np.array(p_img)
        plt.imsave(prototype_fname, p_img)

    except:
        p_img = None


    return p_img



# Function: Save prototype bbox
def save_prototype_box(prototype_bbox_fname, prototypes_img_dir, index):

    try:
        # Note: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
        # Open an image file and save it
        # p_img = plt.imread(os.path.join(load_img_dir, 'prototype-img-with_box'+str(index)+'.png'))
        p_img = Image.open(os.path.join(prototypes_img_dir, 'prototype-img-with_box'+str(index)+'.png')).convert("RGB")
        p_img = np.array(p_img)
        plt.imsave(prototype_bbox_fname, p_img)

    except:
        p_img = None


    return p_img



# Function: Save image with a bounding-box
def imsave_with_bbox(
        img_w_bbox_fname,
        img_rgb,
        bbox_height_start,
        bbox_height_end,
        bbox_width_start,
        bbox_width_end,
        color=(0, 255, 255)):


    try:
        # Convert RGB to BGR with OpenCV
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)

        # Draw rectangle around bbox
        cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)

        # Convert into RGB and type float
        img_rgb_uint8 = img_bgr_uint8[...,::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255
        plt.imsave(img_w_bbox_fname, img_rgb_float)

    except:
        img_rgb_float = None


    return img_rgb_float



# Function: Save deformable prototype information
def save_deform_info(
        model,
        offsets,
        input,
        activations,
        save_dir,
        prototype_img_filename_prefix,
        proto_index,
        prototype_layer_stride=1):

    # Get prototype shape
    prototype_shape = model.prototype_shape

    # FIXME: Review this piece of code
    if not hasattr(model, "prototype_dilation"):
        dilation = model.prototype_dillation
    else:
        dilation = model.prototype_dilation


    # Get input shape
    original_img_size = input.shape[0]

    colors = [
        (230/255, 25/255, 75/255),
        (60/255, 180/255, 75/255),
        (255/255, 225/255, 25/255),
        (0, 130/255, 200/255),
        (245/255, 130/255, 48/255),
        (70/255, 240/255, 240/255),
        (240/255, 50/255, 230/255),
        (170/255, 110/255, 40/255), (0,0,0)
    ]


    # Get Prototype Activations
    argmax_proto_act_j = list(np.unravel_index(np.argmax(activations, axis=None), activations.shape))
    fmap_height_start_index = argmax_proto_act_j[0] * prototype_layer_stride
    fmap_width_start_index = argmax_proto_act_j[1] * prototype_layer_stride


    # Get original image(s) with bboxes
    original_img_j_with_boxes = input.copy()


    # Get number of deformable groups (model.num_prototypes // model.num_classes)
    num_def_groups = 1

    # Get deformable group(s) indices and offset
    def_grp_index = proto_index % num_def_groups
    def_grp_offset = def_grp_index * 2 * prototype_shape[-2] * prototype_shape[-1]


    # Iterate through prototype shape(s) to get the prototypes
    prototypes_just_this_box = list()
    prototypes_patches = list()

    for i in range(prototype_shape[-2]):
        for k in range(prototype_shape[-1]):

            # Offsets go in order height offset, width offset
            h_index = def_grp_offset + 2 * (k + prototype_shape[-2]*i)
            w_index = h_index + 1
            h_offset = offsets[0, h_index, fmap_height_start_index, fmap_width_start_index]
            w_offset = offsets[0, w_index, fmap_height_start_index, fmap_width_start_index]

            # Subtract prototype_shape // 2 because fmap start indices give the center location, and we start with top left
            def_latent_space_row = fmap_height_start_index + h_offset + (i - prototype_shape[-2] // 2) * dilation[0]
            def_latent_space_col = fmap_width_start_index + w_offset + (k - prototype_shape[-1] // 2)* dilation[1]

            # Map the coordinates above into the image under analysis
            def_image_space_row_start = int(def_latent_space_row * original_img_size / activations.shape[-2])
            def_image_space_row_end = int((1 + def_latent_space_row) * original_img_size / activations.shape[-2])
            def_image_space_col_start = int(def_latent_space_col * original_img_size / activations.shape[-1])
            def_image_space_col_end = int((1 + def_latent_space_col) * original_img_size / activations.shape[-1])


            # Get the image with this bbox
            img_with_just_this_box = input.copy()

            # Draw rectangle
            cv2.rectangle(
                img_with_just_this_box,
                (def_image_space_col_start, def_image_space_row_start),
                (def_image_space_col_end, def_image_space_row_end),
                colors[i*prototype_shape[-1] + k],
                1
            )


            # Add this to the corresponding list
            prototypes_just_this_box.append(img_with_just_this_box)

            # Save this image (if needed)
            plt.imsave(
                os.path.join(save_dir, prototype_img_filename_prefix + str(proto_index) + '_patch_' + str(i*prototype_shape[-1] + k) + '-with_box.png'),
                img_with_just_this_box,
                vmin=0.0,
                vmax=1.0
            )


            # Draw rectangle ont the other image with bboxes
            cv2.rectangle(
                original_img_j_with_boxes,
                (def_image_space_col_start, def_image_space_row_start),
                (def_image_space_col_end, def_image_space_row_end),
                colors[i*prototype_shape[-1] + k],
                1
            )


            # Save this image uppon a given condition (defined by the original authors)
            if not (def_image_space_col_start < 0 or def_image_space_row_start < 0 or def_image_space_col_end >= input.shape[0] or def_image_space_row_end >= input.shape[1]):

                # Append this to the list
                prototypes_patches.append(input[def_image_space_row_start:def_image_space_row_end, def_image_space_col_start:def_image_space_col_end, :])
                plt.imsave(
                    os.path.join(save_dir, prototype_img_filename_prefix + str(proto_index) + '_patch_' + str(i*prototype_shape[-1] + k) + '.png'),
                    input[def_image_space_row_start:def_image_space_row_end, def_image_space_col_start:def_image_space_col_end, :],
                    vmin=0.0,
                    vmax=1.0
                )


    plt.imsave(
        os.path.join(save_dir, prototype_img_filename_prefix + str(proto_index) + '-with_box.png'),
        original_img_j_with_boxes,
        vmin=0.0,
        vmax=1.0
    )

    return prototypes_just_this_box, prototypes_patches, original_img_j_with_boxes



# CUB2002011Dataset: Dataset Class
class CUB2002011Dataset(Dataset):
    def __init__(self, data_path, fold=0, split="train", transform=None):

        """
        Args:
            data_path (string): Data directory.
            classes_txt (string): File with the classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        assert split in ("train", "val", "test")
        self.fold = fold
        self.split = split
        self.data_path = data_path


        # Get the path of the split
        splits_df = pd.read_csv(os.path.join(data_path, "processed", "data_splits.csv"))
        dataset = splits_df[(splits_df['fold'] == fold) & (splits_df['split'] == split)]

        # Extract labels from data path
        labels = np.genfromtxt(os.path.join(data_path, "CUB_200_2011", "classes.txt"), dtype=str)
        labels_dict = dict()
        for label_info in labels:
            labels_dict[label_info[1]] = int(label_info[0]) - 1


        # Extract bounding box information
        images = np.genfromtxt(os.path.join(data_path, "CUB_200_2011", "images.txt"), dtype=str)
        bounding_boxes = np.genfromtxt(os.path.join(data_path, "CUB_200_2011", "bounding_boxes.txt"), dtype=str)

        # Create a list with all info
        bbox_dict = dict()
        for img in images:
            for bbox in bounding_boxes:
                img_id = img[0]
                img_fname = img[1]
                bbox_id = bbox[0]
                bbox_coord = bbox[1::]

                if img_id == bbox_id:
                    img_fname_ = img_fname.split("/")[-1]
                    bbox_dict[img_fname_] = bbox_coord


        self.dataset = dataset
        self.labels_dict = labels_dict
        self.bbox_dict = bbox_dict
        self.transform = transform

        return



    # Method: __len__
    def __len__(self):
        return len(self.dataset)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # Get image
        img_path = self.dataset.iloc[idx]['image_fnames']

        # Load image
        image = Image.open(os.path.join(self.data_path, "CUB_200_2011", "images", img_path)).convert('RGB')

        # Load bounding box
        img_bbox = self.bbox_dict[img_path.split('/')[-1]]


        # Get bounding boxes info
        x = float(img_bbox[0])
        y = float(img_bbox[1])
        width = float(img_bbox[2])
        height = float(img_bbox[3])

        # Crop image
        image = image.crop((x, y, x+width, y+height))


        # Get labels
        folder = img_path.split("/")[0]
        label = self.labels_dict[folder]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label



# PH2Dataset: Dataset Class
class PH2Dataset(Dataset):
    def __init__(self, data_path, fold=0, split="train", transform=None):

        """
        Args:
            data_path (string): Data directory.
            fold (int): Fold number.
            split (string): Data split ('train', 'val', 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        assert split in ("train", "val", "test"), "Subset must be in ('train', 'val', 'test')."

        # Get the path of the split
        splits_df = pd.read_csv(os.path.join(data_path, "processed", "data_splits.csv"))
        dataset = splits_df[(splits_df['fold'] == fold) & (splits_df['split'] == split)]

        # Create a diagnosis dictionary
        self.diagnosis_dict = {
            "Common Nevus":0,
            "Atypical Nevus":1,
            "Melanoma":2
        }

        # Create an inverse diagnosis dictionary
        self.inv_diagnosis_dict = {
            0:"Common Nevus",
            1:"Atypical Nevus",
            2:"Melanoma"
        }


        # Labels Dictionary
        labels_dict = dict()
        for img, label in zip(dataset["image_name"].values,dataset["image_label"].values):
            labels_dict[img] = label


        self.data_path = data_path
        self.dataset = dataset
        self.labels_dict = labels_dict
        self.transform = transform


        return



    # Method: __len__
    def __len__(self):
        return len(self.dataset)



    # Method: __getitem__
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load and crop image
        image_name = self.dataset.iloc[idx]['image_name']
        img_path = os.path.join(self.data_path, "images", image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp")
        image = Image.open(img_path).convert('RGB')
        image = self.crop_image(image=image, image_name=image_name)

        # Get labels
        label = self.dataset.iloc[idx]["image_label"]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label


    # Method: Crop image
    def crop_image(self, image, image_name):

        # Open mask
        pil_mask = Image.open(os.path.join(self.data_path, "images", image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp")).convert('L')

        # Convert PIL mask to NumPy array
        npy_mask = np.array(pil_mask.copy())

        # We have to be sure we are working with binary image
        ret, binary = cv2.threshold(npy_mask, 127, 255, cv2.THRESH_BINARY)

        # Get countours of this image
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding box from mask
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop image
        crop_img = image.copy().crop((x, y, x+w, y+h))

        return crop_img



# PAPILADataset: Dataset Class
class PAPILADataset(Dataset):
    def __init__(self, data_path, fold=0, split='train', transform=None):

        """
        Args:
            data_path (string): Data directory.
            fold (int): Fold number.
            split (string): Data split ('train', 'val', 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        assert split in ("train", "val", "test"), "Split must be in ('train', 'val', 'test')."

        # Get the path of the split
        splits_df = pd.read_csv(os.path.join(data_path, "processed", "data_splits.csv"))
        dataset = splits_df[(splits_df['fold'] == fold) & (splits_df['split'] == split)]

        # Labels Dictionary
        labels_dict = dict()
        for img, label in zip(dataset["images_fnames"].values, dataset["images_labels"].values):
            labels_dict[img] = label


        self.data_path = data_path
        self.dataset = dataset
        self.images_labels = dataset["images_labels"].values
        self.labels_dict = labels_dict
        self.transform = transform


        return



    # Method: __len__
    def __len__(self):
        return len(self.dataset)



    # Method: __getitem__
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load and crop image
        image_name = self.dataset.iloc[idx]['images_fnames'].replace(".png", ".jpg")
        image = Image.open(os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "FundusImages", image_name))
        image = self.crop_image(image=image, image_name=image_name)

        # Get labels
        label = self.dataset.iloc[idx]['images_labels']

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label


    # Method: Crop image
    def crop_image(self, image, image_name):

        # Convert to NumPy
        npy_img = np.array(image)

        # Get ROIs (We use both expert annotations)
        cup_path1 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_cup_exp1.txt")
        disc_path1 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_disc_exp1.txt")
        cup_path2 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_cup_exp2.txt")
        disc_path2 = os.path.join(self.data_path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f", "ExpertsSegmentations", "Contours", image_name.split(".")[0] + "_disc_exp2.txt")


        # Generate ROIs
        cup_mask1 = self.contour_to_mask(cup_path1, npy_img.shape)
        disc_mask1 = self.contour_to_mask(disc_path1, npy_img.shape)
        cup_mask2 = self.contour_to_mask(cup_path2, npy_img.shape)
        disc_mask2 = self.contour_to_mask(disc_path2, npy_img.shape)

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
        crop_img = image.crop((x, y, x+w, y+h))
        # plt.imshow(np.array(crop_img), cmap="gray")
        # plt.show()


        # Some debug tests using OpenCV
        # crop_img = npy_img[y:y+h, x:x+w]
        # plt.imshow(np.array(crop_img), cmap="gray")
        # plt.show()

        return crop_img

    # Function: Return mask given a contour and the shape of image
    # contour_to_mask() and apply_mask() functions taken from https://github.com/matterport/Mask_RCNN
    def contour_to_mask(self, contour_txt_path, img_shape):
        """
        Return mask given a contour and the shape of image
        """

        # Get coordinates
        c = np.loadtxt(contour_txt_path)

        # Generate mask
        mask = np.zeros(img_shape[:-1], dtype=np.uint8)
        rr, cc = drw.polygon(c[:,1], c[:,0])
        mask[rr, cc] = 1

        return mask



    # Function: Apply the given mask to the image
    def apply_mask(self, image, mask, color, alpha=0.5):
        """
        Apply the given mask to the image.
        """


        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])


        return image
