# Note: Provide an input image, and get the nearest counterfactual (in the baseline version we can not compare prototypes)



# Imports
import os
import argparse
import pandas as pd
import numpy as np

# PyTorch Imports
import torch
import torch.utils.data
import torchvision

# Project Imports
from data_utilities import CUB2002011Dataset, PAPILADataset, PH2Dataset, STANFORDCARSDataset
from image_retrieval_utilities import generate_image_features, get_image_counterfactual, get_image_prediction
from model_utilities import DenseNet, ResNet, VGG



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PAPILA", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")

# Model
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, densenet161, resnet34, resnet152, vgg16, vgg19.')

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms.")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")

# Get checkpoint
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint that contains weights and model parameters.")

# Generate image's features
parser.add_argument("--generate_img_features", action="store_true", help="Generate features for the retrieval.")

# Decide the type of features to generate and to use in the retrieval
parser.add_argument("--feature_space", type=str, required=True, choices=["conv_features"], help="Feature space: convolutional features (conv_features).")


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

# Image size (after transforms)
IMG_SIZE = args.img_size

# Generate features on the test set
GENERATE_FEATURES = args.generate_img_features

# Feature space
FEATURE_SPACE = args.feature_space



# Get the directory of results
results_dir = os.path.join(OUTPUT_DIR, CHECKPOINT)


# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



# Transforms
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



# Weights
weights_dir = os.path.join(results_dir, "weights")


# Features 
features_dir = os.path.join(results_dir, "features", FEATURE_SPACE)
if not os.path.isdir(features_dir):
    os.makedirs(features_dir)


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Construct the Model
if BASE_ARCHITECTURE.lower() in ("densenet121", "densenet161"):
    baseline_model = DenseNet(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)
elif BASE_ARCHITECTURE.lower() in ("resnet34", "resnet152"):
    baseline_model = ResNet(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)
else:
    baseline_model = VGG(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)



# Put model into DEVICE (CPU or GPU)
baseline_model = baseline_model.to(DEVICE)

# Load model weights (we load the weights that correspond to the last stage of training)
# model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
# model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
model_weights = torch.load(model_path_push_last, map_location=DEVICE)
baseline_model.load_state_dict(model_weights['model_state_dict'], strict=True)
print(f"Model weights loaded with success from: {model_path_push_last}.")


# Put model into evaluation mode
baseline_model.eval()


# Create a local analysis path
save_analysis_path = os.path.join(results_dir, "analysis", "image-retrieval", FEATURE_SPACE)
if not os.path.isdir(save_analysis_path):
    os.makedirs(save_analysis_path)



# Generate images features (we will need these for the retrieval)
# Note: We generate features for the entire database
if GENERATE_FEATURES:

    for directory, data_path, labels_dict in zip([train_img_directories, test_img_directories], [train_data_path, test_data_path], [train_labels_dict, test_labels_dict]):

        # Go through all image directories
        for image_dir in directory:

            # Get images in this directory
            image_names = [i for i in os.listdir(os.path.join(data_path, image_dir)) if not i.startswith('.')]
            image_names = [i for i in image_names if i != "augmented"]
            image_names = [i for i in image_names if not os.path.isdir(os.path.join(data_path, i))]

            # Go through all images in a single directory
            for image_name in image_names:

                # Get image label
                image_label = labels_dict[image_dir]

                # Generate test image path
                image_path = os.path.join(data_path, image_dir, image_name)


                # Generate features
                features = generate_image_features(
                    image_path=image_path,
                    baseline_model=baseline_model,
                    device=DEVICE,
                    transforms=transforms,
                    feature_space=FEATURE_SPACE
                )


                # Convert feature vector to NumPy
                features = features.detach().cpu().numpy()

                # Save this into disk
                features_fname = image_name.split('.')[0] + '.npy'
                features_fpath = os.path.join(features_dir, features_fname)
                np.save(
                    file=features_fpath,
                    arr=features,
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



# Query Image: Here, we will only use the images of the test set as query images
# Go through all test image directories
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


        # Get image counterfactual
        label_pred, counterfactual_pred = get_image_counterfactual(
            image_path=test_img_path,
            baseline_model=baseline_model,
            device=DEVICE,
            transforms=transforms
        )


        # Check if the predicted label is equal to the ground-truth label
        if int(image_label) == int(label_pred):

            # Then we check for counterfactuals
            # First, we get the features, by loading the feature vectors
            if GENERATE_FEATURES:
                test_img_fts = np.load(os.path.join(features_dir, image_name.split('.')[0] + '.npy'), allow_pickle=True, fix_imports=True)
            else:
                test_img_fts = generate_image_features(
                    image_path=test_img_path,
                    baseline_model=baseline_model,
                    device=DEVICE,
                    transforms=transforms,
                    feature_space=FEATURE_SPACE
                )


            # Create lists to append temporary values
            counter_imgs_fnames = list()
            distances = list()
            
            
            # Iterate again through the TRAIN images of the database
            for ctf_directory, ctf_data_path, ctf_labels_dict in zip([train_img_directories], [train_data_path], [train_labels_dict]):
                for counterfact_dir in ctf_directory:

                    # Get images in this directory
                    ctf_names = [i for i in os.listdir(os.path.join(ctf_data_path, counterfact_dir)) if not i.startswith('.')]
                    ctf_names = [i for i in ctf_names if i != "augmented"]
                    ctf_names = [i for i in ctf_names if not os.path.isdir(os.path.join(ctf_data_path, i))]

                    # Get label of the counterfactual first
                    ctf_label = ctf_labels_dict[counterfact_dir]

                    # We only evaluate in such cases
                    if int(ctf_label) == int(counterfactual_pred):
                        for ctf_fname in ctf_names:

                            # Get the prediction of the model on this counterfactual
                            ctf_prediction = get_image_prediction(
                                image_path=os.path.join(ctf_data_path, counterfact_dir, ctf_fname),
                                baseline_model=baseline_model,
                                device=DEVICE,
                                transforms=transforms
                            )


                            # Only compute the distances to cases where both the ground-truth and the predicted label(s) of the counterfactual match
                            if int(ctf_prediction) == int(ctf_label):
                                
                                # Load the features of the counterfactual
                                if GENERATE_FEATURES:
                                    ctf_fts = np.load(os.path.join(features_dir, ctf_fname.split('.')[0] + '.npy'), allow_pickle=True, fix_imports=True)
                                # Or generate the vector, anyway    
                                else:
                                    ctf_fts = generate_image_features(
                                        image_path=os.path.join(ctf_data_path, counterfact_dir, ctf_fname),
                                        baseline_model=baseline_model,
                                        device=DEVICE,
                                        transforms=transforms,
                                        feature_space=FEATURE_SPACE
                                    )

                                    

                                # Compute the Euclidean Distance (L2-norm) between these feature vectors
                                distance_img_ctf = np.linalg.norm(test_img_fts-ctf_fts)


                                # Append these to lists
                                counter_imgs_fnames.append(ctf_fname)
                                distances.append(distance_img_ctf)
    


            # Add information to our data dictionary
            analysis_dict["Image"].append(image_name)
            analysis_dict["Image Label"].append(int(image_label))

            # We must be sure that we found at least one valid counterfactual
            if len(distances) > 0:
                analysis_dict["Nearest Counterfactual"].append(counter_imgs_fnames[np.argmin(distances)])
            else:
                analysis_dict["Nearest Counterfactual"].append("N/A")
            
            analysis_dict["Nearest Counterfactual Label"].append(int(counterfactual_pred))



# Save data dictionary into a .CSV
analysis_df = pd.DataFrame.from_dict(data=analysis_dict)

# Check if old analysis.csv file exists
csv_path = os.path.join(save_analysis_path, "analysis.csv")
if os.path.exists(csv_path):
    os.remove(csv_path)

# Save new file
analysis_df.to_csv(path_or_buf=csv_path)



print("Finished.")
