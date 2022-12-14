# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/local_analysis.py
# Note: Finding the nearest prototypes to a test image

# Imports
import os
import argparse
import pandas as pd

# PyTorch Imports
import torch
import torchvision
import torch.utils.data

# Project Imports
from data_utilities import CUB2002011Dataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
from prototypes_retrieval_utilities import retrieve_image_prototypes



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

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms.")

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")

# Get checkpoint
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint that contains weights and model parameters.")



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

# Image size (after transforms)
IMG_SIZE = args.img_size

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type


# Get the directory of results
results_dir = os.path.join(OUTPUT_DIR, CHECKPOINT)


# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



# Test Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


# Dataset
# CUB2002011
if DATASET == "CUB2002011":

    # Get image directories
    data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped")
    image_directories = [f for f in os.listdir(data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = CUB2002011Dataset(
        data_path=data_path,
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(test_set.labels_dict)

    # Labels dictionary
    labels_dict = test_set.labels_dict.copy()


# PH2
elif DATASET == "PH2":

    # Get image directories
    data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "test", "cropped")
    image_directories = [f for f in os.listdir(data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        augmented=False,
        transform=test_transforms
    )

    # Number of Classes
    NUM_CLASSES = len(test_set.diagnosis_dict)

    # Labels dictionary
    labels_dict = test_set.labels_dict.copy()


# STANFORDCARS
elif DATASET == "STANFORDCARS":

    # Get image directories
    data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_test", "images_cropped")
    image_directories = [f for f in os.listdir(data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_test",
        augmented=False,
        cropped=True,
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(test_set.class_names)

    # Labels dictionary
    labels_dict = test_set.labels_dict.copy()



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


# Weights
weights_dir = os.path.join(results_dir, "weights")

# Prototypes
load_img_dir = os.path.join(weights_dir, 'prototypes')


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Construct the Model
ppnet_model = construct_PPNet(
    base_architecture=BASE_ARCHITECTURE,
    pretrained=False,
    img_size=IMG_SIZE,
    prototype_shape=PROTOTYPE_SHAPE,
    num_classes=NUM_CLASSES,
    prototype_activation_function=PROTOTYPE_ACTIVATION_FUNCTION,
    add_on_layers_type=ADD_ON_LAYERS_TYPE
)


# Define if the model should be class specific
class_specific = True



# Put model into DEVICE (CPU or GPU)
ppnet_model = ppnet_model.to(DEVICE)

# Load model weights (we should read the last optimised layer)
# model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
# model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
model_weights = torch.load(model_path_push_last, map_location=DEVICE)
ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
print(f"Model weights loaded with success from: {model_path_push_last}.")


# Create a local analysis path
save_analysis_path = os.path.join(results_dir, "analysis", "local")
if not os.path.isdir(save_analysis_path):
    os.makedirs(save_analysis_path)


# Get prototype shape
prototype_shape = ppnet_model.prototype_shape

# Get max distance
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]



# Analysis .CSV
analysis_dict = {
    "Image Filename":list(),
    "Ground-Truth Label":list(),
    "Predicted Label":list(),
    "Number of Prototypes Connected to the Class Identity":list(),
    "Top-10 Prototypes Class Identities":list()
}


# Go through all image directories
for image_dir in image_directories:

    # Get images in this directory
    image_names = [i for i in os.listdir(os.path.join(data_path, image_dir)) if not i.startswith('.')]
    image_names = [i for i in image_names if not os.path.isdir(os.path.join(data_path, i))]

    # Go through all images in a single directory
    for image_name in image_names:

        # Get image label
        image_label = labels_dict[image_dir]

        # Create image analysis path
        image_analysis_path = os.path.join(save_analysis_path, image_dir, image_name.split('.')[0])
        if not os.path.isdir(image_analysis_path):
            os.makedirs(image_analysis_path)

        # Analyse this image
        img_fname, gt_label, pred_label, nr_prototypes_cls_ident, topk_proto_cls_ident = retrieve_image_prototypes(
            save_analysis_path=image_analysis_path,
            weights_dir=weights_dir,
            load_img_dir=load_img_dir,
            ppnet_model=ppnet_model,
            device=DEVICE,
            test_transforms=test_transforms,
            test_image_dir=os.path.join(data_path, image_dir),
            test_image_name=image_name,
            test_image_label=image_label,
            norm_params={"mean":MEAN, "std":STD},
            img_size=IMG_SIZE
        )


        # Add information to our data dictionary
        analysis_dict["Image Filename"].append(img_fname)
        analysis_dict["Ground-Truth Label"].append(gt_label)
        analysis_dict["Predicted Label"].append(pred_label)
        analysis_dict["Number of Prototypes Connected to the Class Identity"].append(nr_prototypes_cls_ident)
        analysis_dict["Top-10 Prototypes Class Identities"].append(topk_proto_cls_ident)


# Save data dictionary into a .CSV
analysis_df = pd.DataFrame.from_dict(data=analysis_dict)
analysis_df.to_csv(path_or_buf=os.path.join(save_analysis_path, "analysis.csv"))



print("Finished.")
