# Note: Provide an input image, get the nearest counterfactual and compare prototypes



# Imports
import os
import argparse
import pandas as pd
import numpy as np

# PyTorch Imports
import torch
import torch.utils.data
from torch.autograd import Variable
import torchvision

# Project Imports
from data_utilities import CUB2002011Dataset, PH2Dataset, STANFORDCARSDataset
from image_retrieval_utilities import generate_image_features, get_image_counterfactual
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
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, densenet161, resnet34, resnet152, vgg16, vgg19.')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation.")

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

# Compute metrics on test
parser.add_argument("--compute_metrics", action="store_true", help="Compute metrics on a specific data subset.")

# Generate test images' features
parser.add_argument("--generate_img_features", action="store_true", help="Generate features for the retrieval.")



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

# Compute metrics on data subset
COMPUTE_METRICS = args.compute_metrics

# Generate features on the test set
GENERATE_FEATURES = args.generate_img_features



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



# PH2
elif DATASET == "PH2":

    # Get train image directories
    train_data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "train", "cropped")
    train_img_directories = [f for f in os.listdir(train_data_path) if not f.startswith('.')]

    # Train Dataset
    train_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="test",
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



# Create Test DataLoader
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)


# Weights
weights_dir = os.path.join(results_dir, "weights")


# Features 
features_dir = os.path.join(results_dir, "features", "test")
if not os.path.isdir(features_dir):
    os.makedirs(features_dir)


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

# Load model weights (we load the weights that correspond to the last stage of training)
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


# Get model performance metrics
# accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=print)
if COMPUTE_METRICS:
    metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
    test_accuracy = metrics_dict["accuracy"]
    print(f"Accuracy on test: {test_accuracy}.")



# Generate images features (we will need these for the retrieval)
if GENERATE_FEATURES:

    # Go through all image directories
    for image_dir in test_img_directories:

        # Get images in this directory
        image_names = [i for i in os.listdir(os.path.join(test_data_path, image_dir)) if not i.startswith('.')]
        image_names = [i for i in image_names if not os.path.isdir(os.path.join(test_data_path, i))]

        # Go through all images in a single directory
        for image_name in image_names:

            # Get image label
            image_label = test_labels_dict[image_dir]

            # Generate test image path
            test_img_path = os.path.join(test_data_path, image_dir, image_name)


            # Generate features
            conv_features = generate_image_features(
                image_path=test_img_path,
                ppnet_model=ppnet_model,
                device=DEVICE,
                transforms=transforms
            )


            # Convert feature vector to NumPy
            conv_features = conv_features.detach().cpu().numpy()

            # Save this into disk
            conv_features_fname = image_name.split('.')[0] + '.npy'
            conv_features_fpath = os.path.join(features_dir, conv_features_fname)
            np.save(
                file=conv_features_fpath,
                arr=conv_features,
                allow_pickle=True,
                fix_imports=True
            )



# Analysis .CSV
analysis_dict = {
    "Image":list(),
    "Image Label":list(),
    "Nearest Counterfactual":list(),
    "Nearest Counterfactual Label":list(),
}



# Go through all image directories
for image_dir in test_img_directories:

    # Get images in this directory
    image_names = [i for i in os.listdir(os.path.join(test_data_path, image_dir)) if not i.startswith('.')]
    image_names = [i for i in image_names if not os.path.isdir(os.path.join(test_data_path, i))]

    # Go through all images in a single directory
    for image_name in image_names:

        # Get image label
        image_label = test_labels_dict[image_dir]

        # Generate test image path
        test_img_path = os.path.join(test_data_path, image_dir, image_name)


        


        # Add information to our data dictionary
        analysis_dict["Image Filename"].append(img_fname)
        analysis_dict["Ground-Truth Label"].append(gt_label)
        analysis_dict["Predicted Label"].append(pred_label)
        analysis_dict["Number Prototypes Connected Class Identity"].append(nr_prototypes_cls_ident)
        analysis_dict["Top-10 Activated Prototypes"].append(topk_proto_cls_ident)


# Save data dictionary into a .CSV
analysis_df = pd.DataFrame.from_dict(data=analysis_dict)
analysis_df.to_csv(path_or_buf=os.path.join(save_analysis_path, "analysis.csv"))



print("Finished.")
