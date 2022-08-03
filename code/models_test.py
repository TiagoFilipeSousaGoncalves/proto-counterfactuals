# Imports
import os
import argparse
import numpy as np
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from data_utilities import preprocess_input_function, CUB2002011Dataset
from model_utilities import construct_PPNet
from prototypes_utilities import push_prototypes
from train_val_test_utilities import model_test



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011"], help="Data set: CUB2002011")

# Model
# base_architecture = 'vgg19'
parser.add_argument('--base_architecture', type=str, required=True, choices=["vgg19"], help='Base architecture: vgg19, ')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")

# Prototype shape
# prototype_shape = (2000, 128, 1, 1)
parser.add_argument('--prototype_shape', type=tuple, default=(2000, 128, 1, 1), help="Prototype shape.")

# Number of classes
# num_classes = 200
parser.add_argument('--num_classes', type=int, default=200, help="Number of classes.")

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

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

# Number of classes
NUM_CLASSES = args.num_classes

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type

# Resize (data transforms)
RESIZE_OPT = args.resize



# Get the directory of results
results_dir = os.path.join(OUTPUT_DIR, CHECKPOINT)


# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE




# Test Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


# Dataset
if DATASET == "CUB2002011":
    # Test Dataset
    test_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub_200_2011", "processed_data", "test", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub_200_2011", "source_data", "classes.txt"),
        transform=test_transforms
    )


# Results and Weights
weights_dir = os.path.join(results_dir, "weights")


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


# Test DataLoader
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=WORKERS)


# Test Phase 
print("Test Phase")
print(f'Size of test set: {len(test_loader.dataset)}')
print(f'Batch size: {BATCH_SIZE}')
acc = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)



print("Finished.")
