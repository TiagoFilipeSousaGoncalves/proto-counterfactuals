# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/global_analysis.py
# Note: Finding the nearest patches to each prototype

# Imports
import os
import argparse
import numpy as np

# PyTorch Imports
import torch
import torch.utils.data
import torchvision

# Project Imports
from analysis_utilities import find_k_nearest_patches_to_prototypes, save_prototype_original_img_with_bbox
from data_utilities import preprocess_input_function, CUB2002011Dataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet



# TODO: Erase uppon review
# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
# parser.add_argument('-modeldir', nargs=1, type=str)
# parser.add_argument('-model', nargs=1, type=str)
# parser.add_argument('-dataset', nargs=1, type=str, default='cub200')



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PH2, STANFORDCARS.")

# Model
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, resnet18, vgg19.')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")

# Prototype shape
# prototype_shape = (2000, 128, 1, 1)
# parser.add_argument('--prototype_shape', type=tuple, default=(2000, 128, 1, 1), help="Prototype shape.")

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Get checkpoint
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from which to resume training")

# K-Nearest Patches
parser.add_argument("--k_nearest_patches", type=int, default=5, help="The number of K-nearest patches.")



# Parse the arguments
args = parser.parse_args()



# Get checkpoint (that contains the weights)
CHECKPOINT = args.checkpoint

# Data directory
DATA_DIR = args.data_dir

# Dataset
DATASET = args.dataset

# Base Architecture
BASE_ARCHITECTURE = args.base_architecture

# Results Directory
OUTPUT_DIR = args.output_dir

# Number of workers (threads)
WORKERS = args.num_workers

# Prototype activation function
PROTOTYPE_ACTIVATION_FUNCTION = args.prototype_activation_function

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.img_size

# Prototype shape
PROTOTYPE_SHAPE = args.prototype_shape

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type

# Get the directory of results
results_dir = os.path.join(OUTPUT_DIR, CHECKPOINT)

# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Number of Workers
WORKERS = args.num_workers

# Top-K Patches
K = args.k_nearest_patches



# Load the data
# Note: Must use unaugmented (original) dataset
# train_dir = train_push_dir

# Train Transforms (without normalization)
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
])


# Test Transforms (without normalization)
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
])



# Dataset
# CUB2002011
if DATASET == "CUB2002011":
    # Train Dataset
    train_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=True,
        transform=train_transforms
    )

    # Test Dataset
    test_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.labels_dict)


# PH2
elif DATASET == "PH2":
    # Train Dataset
    train_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="train",
        cropped=True,
        transform=train_transforms
    )

    # Test Dataset
    test_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        transform=test_transforms
    )

    # Number of Classes
    NUM_CLASSES = len(train_set.diagnosis_dict)


# STANFORDCARS
elif DATASET == "STANFORDCARS":
    # Train Dataset
    train_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_train",
        augmented=True,
        cropped=True,
        transform=train_transforms
    )

    # Test Dataset
    test_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_test",
        augmented=False,
        cropped=True,
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.class_names)



# Define prototype shape according to original paper
# The number of prototypes can be chosen with prior domain knowledge or hyperparameter search: we used 10 prototypes per class
NUM_PROTOTYPES_CLASS = int(NUM_CLASSES * 10)

# For VGG-16, VGG-19, DenseNet-121, DenseNet-161, we used 128 as the number of channels in a prototype
if BASE_ARCHITECTURE.lower() in ("densenet121", "densenet161", "vgg16", "vgg19"):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 128, 1, 1)

# For ResNet-34, we used 256 as the number of channels in a prototype;
elif BASE_ARCHITECTURE.lower() in ("resnet34"):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 256, 1, 1)

# For ResNet-152, we used 512 as the number of channels in a prototype
elif BASE_ARCHITECTURE.lower() in ("resnet152"):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 512, 1, 1)



# TODO: Erase uppon review
# load_model_dir = args.modeldir[0]
# load_model_name = args.model[0]

# load_model_path = os.path.join(load_model_dir, load_model_name)
# epoch_number_str = re.search(r'\d+', load_model_name).group(0)
# start_epoch_number = int(epoch_number_str)



# Results and Weights
weights_dir = os.path.join(results_dir, "weights")



# Construct the Model
ppnet_model = construct_PPNet(
    base_architecture=BASE_ARCHITECTURE,
    pretrained=False,
    img_size=IMG_SIZE,
    prototype_shape=PROTOTYPE_SHAPE,
    num_classes=NUM_CLASSES,
    prototype_activation_function=PROTOTYPE_ACTIVATION_FUNCTION,
    add_on_layers_type=ADD_ON_LAYERS_TYPE)


# Define if the model should be class specific
class_specific = True


# Put model into DEVICE (CPU or GPU)
ppnet_model = ppnet_model.to(DEVICE)


# Load model weights
model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
model_weights = torch.load(model_path, map_location=DEVICE)
ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
print("Model weights loaded with success.")



# DataLoaders
# Train
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=False)

# Test
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=False)



# Root directory for saving train images
root_dir_for_saving_train_images = os.path.join(weights_dir, f'K_{K}_nearest_train')
if not os.path.isdir(root_dir_for_saving_train_images):
    os.makedirs(root_dir_for_saving_train_images)


# Root directory for saving test images
root_dir_for_saving_test_images = os.path.join(weights_dir, f'K_{K}_nearest_test')
if not os.path.isdir(root_dir_for_saving_test_images):
    os.makedirs(root_dir_for_saving_test_images)


# Save prototypes in original images
# load_img_dir = os.path.join(load_model_dir, 'img')
# prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'bb'+str(start_epoch_number)+'.npy'))
load_img_dir = os.path.join(weights_dir, 'prototypes')
prototype_info = np.load(os.path.join(load_img_dir, 'bb.npy'))


for j in range(ppnet_model.num_prototypes):
    if not os.path.isdir(os.path.join(root_dir_for_saving_train_images, str(j))):
        os.makedirs(os.path.join(root_dir_for_saving_train_images, str(j)))


    if not os.path.isdir(os.path.join(root_dir_for_saving_test_images, str(j))):
        os.makedirs(os.path.join(root_dir_for_saving_test_images, str(j)))


    save_prototype_original_img_with_bbox(
        fname=os.path.join(root_dir_for_saving_train_images, str(j), 'prototype_in_original_pimg.png'),
        load_img_dir=load_img_dir,
        epoch=None,
        index=j,
        bbox_height_start=prototype_info[j][1],
        bbox_height_end=prototype_info[j][2],
        bbox_width_start=prototype_info[j][3],
        bbox_width_end=prototype_info[j][4],
        color=(0, 255, 255)
    )


    save_prototype_original_img_with_bbox(
        fname=os.path.join(root_dir_for_saving_test_images, str(j), 'prototype_in_original_pimg.png'),
        load_img_dir=load_img_dir,
        epoch=None,
        index=j,
        bbox_height_start=prototype_info[j][1],
        bbox_height_end=prototype_info[j][2],
        bbox_width_start=prototype_info[j][3],
        bbox_width_end=prototype_info[j][4],
        color=(0, 255, 255)
    )



# Find K-Nearest Patches (Train) 
# (in original code K -> K+1)
find_k_nearest_patches_to_prototypes(
        dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_model, # pytorch network with prototype_vectors
        device=DEVICE,
        k=K,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
)


# Find K-Nearest Patches (Test)
find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_model, # pytorch network with prototype_vectors
        device=DEVICE,
        k=K,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
)

print("Finished.")
