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
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PAPILA", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")

# Checkpoint
parser.add_argument('--checkpoint', type=str, default="data", help="Path to the model checkpoint.")

# Decide the type of features to generate and to use in the retrieval
parser.add_argument("--feature_space", type=str, required=True, choices=["conv_features"], help="Feature space: convolutional features (conv_features).")



# Parse the arguments
args = parser.parse_args()



# Get the data directory
DATA_DIR = args.data_dir

# Get the dataset
DATASET = args.dataset

# Get the checkpoint
CHECKPOINT = args.checkpoint

# Feature space
FEATURE_SPACE = args.feature_space



# Dataset
# CUB2002011
if DATASET == "CUB2002011":

    # Get train image path
    train_data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped")

    # Get test image path
    test_data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped")


# PAPILA
elif DATASET == "PAPILA":

    # Get train image path
    train_data_path = os.path.join(DATA_DIR, "papila", "processed", "splits", "train")

    # Get test image path
    test_data_path = os.path.join(DATA_DIR, "papila", "processed", "splits", "test")


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
image_retrieval_df = pd.read_csv(filepath_or_buffer=os.path.join("results", CHECKPOINT, "analysis", "image-retrieval", FEATURE_SPACE, "analysis.csv"), sep=",", header=0)



# Create a new directory to save the results from this analysis
counterfact_exp_dir = os.path.join("results", CHECKPOINT, "analysis", "counterfac-exp", FEATURE_SPACE)
if not os.path.isdir(counterfact_exp_dir):
    os.makedirs(counterfact_exp_dir)



# Open file to append errors
if os.path.exists(os.path.join(counterfact_exp_dir, "err_report.txt")):
    os.remove(os.path.join(counterfact_exp_dir, "err_report.txt"))

err_report = open(os.path.join(counterfact_exp_dir, "err_report.txt"), "at")

# Iterate through the rows of the image_retrieval_df
for index, row in image_retrieval_df.iterrows():

    # Get query image and label
    query_img_fname = row["Image"]
    query_img_label = row["Image Label"]
    query_img = Image.open(os.path.join(test_data_path, query_img_fname.split('.')[0], query_img_fname)).convert("RGB")
    query_img = query_img.resize((224, 224))
    query_img = np.array(query_img)


    # Get counterfactual (not all images of all models have counterfactuals)
    try:
        counterfact_img_fname = row["Nearest Counterfactual"]
        counterfact_label = row["Nearest Counterfactual Label"]
        counterfact_img_path = os.path.join(test_data_path, counterfact_img_fname.split('.')[0], counterfact_img_fname) if os.path.exists(os.path.join(test_data_path, counterfact_img_fname.split('.')[0], counterfact_img_fname)) else os.path.join(train_data_path, counterfact_img_fname.split('.')[0], counterfact_img_fname)
        counterfact_img = Image.open(counterfact_img_path).convert("RGB")
        counterfact_img = counterfact_img.resize((224, 224))
        counterfact_img = np.array(counterfact_img)



        # Plot Query Image and its Counterfactual
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

        # Query Image
        ax1.imshow(query_img, cmap="gray")
        ax1.set_title(f'Query: {query_img_fname} ({query_img_label})')
        ax1.axis("off")


        # Counterfactual
        ax2.imshow(counterfact_img, cmap="gray")
        ax2.set_title(f'Counterfactual: {counterfact_img_fname} ({counterfact_label})')
        ax2.axis("off")


        # Create a directory to save results
        save_path = os.path.join(counterfact_exp_dir, query_img_fname.split('.')[0])
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        
        # Save results
        # plt.show(block=True)
        plt.savefig(fname=os.path.join(counterfact_exp_dir, query_img_fname.split('.')[0], "query_vs_cntf_global.png"))
        plt.close()



    except:
        err_report.write(f"Proper counterfactual not available for {query_img_fname}\n")



# Close error report
err_report.close()

print("Finished.")
