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
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PAPILA", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")

# Model
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, resnet18, vgg19.')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

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

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type



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


# Load model weights
model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")



# Create a report to append these results
if os.path.exists(os.path.join(results_dir, "acc_report.txt")):
    os.remove(os.path.join(results_dir, "acc_report.txt"))

report = open(os.path.join(results_dir, "acc_report.txt"), "at")


# Iterate through these model weights types
for model_fname in [model_path, model_path_push, model_path_push_last]:
    model_weights = torch.load(model_fname, map_location=DEVICE)
    ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
    report.write(f"Model weights loaded with success from {model_fname}.\n")


    # Get performance metrics
    train_metrics_dict = model_test(model=ppnet_model, dataloader=train_loader, device=DEVICE, class_specific=class_specific)
    train_acc = train_metrics_dict["accuracy"]
    report.write(f"Train Accuracy: {train_acc}\n")

    test_metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
    test_accuracy = test_metrics_dict["accuracy"]
    report.write(f"Test Accuracy: {test_accuracy}\n")



print("Finished.")
