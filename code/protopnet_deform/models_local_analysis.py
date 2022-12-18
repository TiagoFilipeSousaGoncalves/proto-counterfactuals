# Imports
import os
import argparse
import numpy as np
import pandas as pd

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from data_utilities import CUB2002011Dataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
from prototypes_retrieval_utilities import retrieve_image_prototypes
from train_val_test_utilities import model_test



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PH2, STANFORDCARS.")

# Model
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet50", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, densenet161, resnet34, resnet50, resnet152, vgg16, vgg19.')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")

# Margin (default=0.1)
parser.add_argument('--margin', type=float, default=0.1)

# Subtractive margin
# subtractive_margin = True (default: True)
parser.add_argument('--subtractive_margin', action="store_true")

# Using deformable convolution (default: True)
parser.add_argument('--using_deform', action="store_true")

# Top-K (default=1)
parser.add_argument('--topk_k', type=int, default=1)

# Deformable Convolution Hidden Channels (default=128)
parser.add_argument('--deformable_conv_hidden_channels', type=int, default=128)

# Number of Prototypes (default=1200)
parser.add_argument('--num_prototypes', type=int, default=1200)

# Dilation
parser.add_argument('--dilation', type=float, default=2)

# Incorrect class connection (default=-0.5)
parser.add_argument('--incorrect_class_connection', type=float, default=-0.5)

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Last layer fixed
# last_layer_fixed = True (default: True)
parser.add_argument('--last_layer_fixed', action="store_true")

# Class Weights (default: False)
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Checkpoint
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from which to resume training")

# Compute metrics on test
parser.add_argument("--compute_metrics", action="store_true", help="Compute metrics on a specific data subset.")


# TODO: Erase uppon review
# prototype_layer_stride = 1



# Parse the arguments
args = parser.parse_args()



# Read checkpoint
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

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.img_size

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type

# Last Layer Fixed
LAST_LAYER_FIXED = args.last_layer_fixed

# Margin
MARGIN = args.margin

# Using subtractive margin
SUBTRACTIVE_MARGIN = args.subtractive_margin

# Using deformable convolution
USING_DEFORM = args.using_deform

# TOPK_K
TOPK_K = args.topk_k

# Deformable convolutional hidden channels
DEFORMABLE_CONV_HIDDEN_CHANNELS = args.deformable_conv_hidden_channels

# Number of Prototypes
NUM_PROTOTYPES = args.num_prototypes

# Dilation
DILATION = args.dilation

# Incorrect class connection
INCORRECT_CLASS_CONNECTION = args.incorrect_class_connection

# Compute Metrics
COMPUTE_METRICS = args.compute_metrics



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



# Test DataLoader
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)
print(f'Size of test set: {len(test_loader.dataset)}')
print(f'Batch size: {BATCH_SIZE}')



# Weights
weights_dir = os.path.join(results_dir, "weights")

# Prototypes
load_img_dir = os.path.join(weights_dir, 'prototypes')


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")



# Define the number of prototypes per class
# if NUM_PROTOTYPES == -1:
NUM_PROTOTYPES_CLASS = NUM_PROTOTYPES

if BASE_ARCHITECTURE.lower() == 'resnet34':
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 512, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

elif BASE_ARCHITECTURE.lower() in ('resnet50', 'resnet152'):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 2048, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

elif BASE_ARCHITECTURE.lower() == 'densenet121':
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 1024, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

elif BASE_ARCHITECTURE.lower() == 'densenet161':
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 2208, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

else:
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 512, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'


# Debug print
print("Add on layers type: ", ADD_ON_LAYERS_TYPE)




# Construct the model
ppnet_model = construct_PPNet(
    device=DEVICE,
    base_architecture=BASE_ARCHITECTURE.lower(),
    pretrained=True,
    img_size=IMG_SIZE,
    prototype_shape=PROTOTYPE_SHAPE,
    num_classes=NUM_CLASSES,
    topk_k=TOPK_K,
    m=MARGIN,
    add_on_layers_type=ADD_ON_LAYERS_TYPE,
    using_deform=USING_DEFORM,
    incorrect_class_connection=INCORRECT_CLASS_CONNECTION,
    deformable_conv_hidden_channels=DEFORMABLE_CONV_HIDDEN_CHANNELS,
    prototype_dilation=2
)
class_specific = True



# Put model into DEVICE (CPU or GPU)
ppnet_model = ppnet_model.to(DEVICE)


# Load model weights
# model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
# model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")

# Load model weights
model_weights = torch.load(model_path_push, map_location=DEVICE)
ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
print(f"Model weights loaded with success from: {model_path_push}.")





# Create a local analysis path
save_analysis_path = os.path.join(results_dir, "analysis", "local")
if not os.path.isdir(save_analysis_path):
    os.makedirs(save_analysis_path)


# Get prototype shape
prototype_shape = ppnet_model.prototype_shape

# Get max distance
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]


# Get model performance metrics
# accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=print)
if COMPUTE_METRICS:
    metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
    test_accuracy = metrics_dict["accuracy"]
    print(f"Accuracy on test: {test_accuracy}.")



# Analysis .CSV
analysis_dict = {
    "Image Filename":list(),
    "Ground-Truth Label":list(),
    "Predicted Label":list(),
    "Number Prototypes Connected Class Identity":list(),
    "Top-10 Activated Prototypes":list()
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
        analysis_dict["Number Prototypes Connected Class Identity"].append(nr_prototypes_cls_ident)
        analysis_dict["Top-10 Activated Prototypes"].append(topk_proto_cls_ident)


# Save data dictionary into a .CSV
analysis_df = pd.DataFrame.from_dict(data=analysis_dict)
analysis_df.to_csv(path_or_buf=os.path.join(save_analysis_path, "analysis.csv"))
