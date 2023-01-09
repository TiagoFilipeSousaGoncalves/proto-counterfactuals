# Imports
import os
import argparse
import numpy as np

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
from data_utilities import CUB2002011Dataset, PAPILADataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
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



# Get the directory of results
results_dir = os.path.join(OUTPUT_DIR, CHECKPOINT)


# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



# Test Transforms
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])



# Dataset
# CUB2002011
if DATASET == "CUB2002011":

    # Get train image directories
    train_data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped")
    train_img_directories = [f for f in os.listdir(train_data_path) if not f.startswith('.')]

    # Train Dataset
    train_set = CUB2002011Dataset(
        data_path=train_data_path,
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=transforms
    )

    # Get train labels dictionary
    train_labels_dict = train_set.labels_dict.copy()


    # Get test image directories
    test_data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped")
    test_img_directories = [f for f in os.listdir(test_data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = CUB2002011Dataset(
        data_path=test_data_path,
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=transforms
    )

    # Get test labels dictionary
    test_labels_dict = test_set.labels_dict.copy()

    # Number of classes
    NUM_CLASSES = len(test_set.labels_dict)



# PAPILA
elif DATASET == "PAPILA":

    # Get train image directories
    train_data_path = os.path.join(DATA_DIR, "papila", "processed", "splits", "train")
    train_img_directories = [f for f in os.listdir(train_data_path) if not f.startswith('.')]


    # Train Dataset
    train_set = PAPILADataset(
        data_path=DATA_DIR,
        subset="train",
        cropped=True,
        augmented=False,
        transform=transforms
    )


    # Get train Labels dictionary
    train_labels_dict = train_set.labels_dict.copy()


    # Get image directories
    test_data_path = os.path.join(DATA_DIR, "papila", "processed", "splits", "test")
    test_img_directories = [f for f in os.listdir(test_data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = PAPILADataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        augmented=False,
        transform=transforms
    )

    # Number of classes
    NUM_CLASSES = len(np.unique(test_set.images_labels))

    # Labels dictionary
    test_labels_dict = test_set.labels_dict.copy()



# PH2
elif DATASET == "PH2":

    # Get train image directories
    train_data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "train", "cropped")
    train_img_directories = [f for f in os.listdir(train_data_path) if not f.startswith('.')]

    # Train Dataset
    train_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="train",
        cropped=True,
        augmented=False,
        transform=transforms
    )

    # Get train Labels dictionary
    train_labels_dict = train_set.labels_dict.copy()


    # Get test image directories
    test_data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "test", "cropped")
    test_img_directories = [f for f in os.listdir(test_data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        augmented=False,
        transform=transforms
    )

    # Get test labels dictionary
    test_labels_dict = test_set.labels_dict.copy()

    # Number of classes
    NUM_CLASSES = len(test_set.diagnosis_dict)


# STANFORDCARS
elif DATASET == "STANFORDCARS":

    # Get train image directories
    train_data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_train", "images_cropped")
    train_img_directories = [f for f in os.listdir(train_data_path) if not f.startswith('.')]

    # Train Dataset
    train_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_train",
        augmented=False,
        cropped=True,
        transform=transforms
    )

    # Train labels dictionary
    train_labels_dict = train_set.labels_dict.copy()


    # Get test image directories
    test_data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_test", "images_cropped")
    test_img_directories = [f for f in os.listdir(test_data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_test",
        augmented=False,
        cropped=True,
        transform=transforms
    )

    # Test labels dictionary
    test_labels_dict = test_set.labels_dict.copy()


    # Number of classes
    NUM_CLASSES = len(test_set.class_names)



# DataLoaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)
# print(f'Size of test set: {len(test_loader.dataset)}')
# print(f'Batch size: {BATCH_SIZE}')



# Results and Weights
weights_dir = os.path.join(results_dir, "weights")


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")



# Define the number of prototypes per class
# if NUM_PROTOTYPES == -1:
# NUM_PROTOTYPES_CLASS = NUM_PROTOTYPES
NUM_PROTOTYPES_CLASS = int(NUM_CLASSES * 10)

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
model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")

if not LAST_LAYER_FIXED:
    model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
else:
    model_path_push_last = None


# Create a report to append these results
if os.path.exists(os.path.join(results_dir, "acc_report.txt")):
    os.remove(os.path.join(results_dir, "acc_report.txt"))

report = open(os.path.join(results_dir, "acc_report.txt"), "at")


# Iterate through these model weights types
for model_fname in [model_path, model_path_push, model_path_push_last]:

    if model_fname is not None:
        model_weights = torch.load(model_fname, map_location=DEVICE)
        ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
        report.write(f"Model weights loaded with success from {model_fname}.\n")


        # Get performance metrics
        metrics_dict = model_test(model=ppnet_model, dataloader=train_loader, device=DEVICE, class_specific=class_specific)
        test_accuracy = metrics_dict["accuracy"]
        report.write(f"Train Accuracy: {test_accuracy}\n")

        metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
        test_accuracy = metrics_dict["accuracy"]
        report.write(f"Test Accuracy: {test_accuracy}\n")



print("Finished.")
