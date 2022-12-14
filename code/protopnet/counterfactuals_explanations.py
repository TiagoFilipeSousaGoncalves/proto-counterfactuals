# Imports
import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PH2, STANFORDCARS.")


# Checkpoint
parser.add_argument('--checkpoint', type=str, default="data", help="Path to the model checkpoint.")



# Parse the arguments
args = parser.parse_args()



# Get the data directory
DATA_DIR = args.data_dir

# Get the dataset
DATASET = args.dataset

# Get the checkpoint
CHECKPOINT = args.checkpoint



# Dataset
# CUB2002011
if DATASET == "CUB2002011":

    # Get train image path
    train_data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped")

    # Get test image path
    test_data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped")


# PH2
elif DATASET == "PH2":

    # Get train image path
    train_data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "train", "cropped")

    # Get test image path
    test_data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "test", "cropped")


# STANFORDCARS
elif DATASET == "STANFORDCARS":

    # Get train image directories
    train_data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_train", "images_cropped")
    
    # Get test image directories
    test_data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_test", "images_cropped")



# Get pair(s) "image-counterfactuals" .CSV file
image_retrieval_df = pd.read_csv(filepath_or_buffer=os.path.join("results", CHECKPOINT, "analysis", "image-retrieval", "analysis.csv"), sep=",", header=0)

# Get the prototype statistics .CSV file
proto_stats_df = pd.read_csv(filepath_or_buffer=os.path.join("results", CHECKPOINT, "analysis", "local", "analysis.csv"), sep=",", header=0)



# Iterate through the rows of the image_retrieval_df
for index, row in image_retrieval_df.iterrows():

    # Get query image and label
    query_img_fname = row["Image"]
    query_img_label = row["Image Label"]
    query_img = Image.open(os.path.join(test_data_path, query_img_fname.split('.')[0], query_img_fname)).convert("RGB")
    query_img = query_img.resize((224, 224))
    query_img = np.array(query_img)


    # Get counterfactual
    counterfact_img_fname = row["Nearest Counterfactual"]
    counterfact_label = row["Nearest Counterfactual Label"]
    counterfact_img = Image.open(os.path.join(test_data_path, counterfact_img_fname.split('.')[0], counterfact_img_fname)).convert("RGB")
    counterfact_img = counterfact_img.resize((224, 224))
    counterfact_img = np.array(counterfact_img)



    # Explanation
    print("Why?")
    print(f'Class Identities of the Prototypes Activated by Query Image: {proto_stats_df[proto_stats_df["Image Filename"]==query_img_fname]["Top-10 Prototypes Class Identities"].values}')
    print(f'Class Identities of the Prototypes Activated by Counterfactual Image: {proto_stats_df[proto_stats_df["Image Filename"]==counterfact_img_fname]["Top-10 Prototypes Class Identities"].values}')



    # Four axes, returned as a 2-d array
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Query Image
    ax1.imshow(query_img, cmap="gray")
    ax1.set_title(f'Query Image (Label {query_img_label})')
    ax1.axis("off")


    # Counterfactual
    ax2.imshow(counterfact_img, cmap="gray")
    ax2.set_title(f'Counterfactual (Label {counterfact_label})')
    ax2.axis("off")


    plt.show(block=True)
