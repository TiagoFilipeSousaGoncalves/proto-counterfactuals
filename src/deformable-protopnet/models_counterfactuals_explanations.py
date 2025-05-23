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
parser.add_argument("--feature_space", type=str, required=True, choices=["conv_features", "proto_features"], help="Feature space: convolutional features (conv_features) or prototype layer features (proto_features).")



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
    test_img_directories = [f for f in os.listdir(test_data_path) if not f.startswith('.')]

    # We need to map these directories into the names of the classes
    test_map_dirs_dict = dict()
    for img_dir in test_img_directories:
        test_map_dirs_dict[img_dir.split('.')[1].lower()] = img_dir


    # Get the number of classes of this dataset
    N_CLASSES = 200


# PAPILA
elif DATASET == "PAPILA":

    # Get train image path
    train_data_path = os.path.join(DATA_DIR, "papila", "processed", "splits", "train")

    # Get test image path
    test_data_path = os.path.join(DATA_DIR, "papila", "processed", "splits", "test")

    # Get the number of classes of this dataset
    N_CLASSES = 3


# PH2
elif DATASET == "PH2":

    # Get train image path
    train_data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "train", "cropped")

    # Get test image path
    test_data_path = os.path.join(DATA_DIR, "ph2", "processed_images", "test", "cropped")

    # Get the number of classes of this dataset
    N_CLASSES = 3


# STANFORDCARS
elif DATASET == "STANFORDCARS":

    # Get train image directories
    train_data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_train", "images_cropped")
    
    # Get test image directories
    test_data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_test", "images_cropped")



# Get pair(s) "image-counterfactuals" .CSV file
image_retrieval_df = pd.read_csv(filepath_or_buffer=os.path.join("results", CHECKPOINT, "analysis", "image-retrieval", FEATURE_SPACE, "analysis.csv"), sep=",", header=0)

# Get the prototype statistics .CSV file
proto_stats_df = pd.read_csv(filepath_or_buffer=os.path.join("results", CHECKPOINT, "analysis", "local", "analysis.csv"), sep=",", header=0)



# Create a new directory to save the results from this analysis
counterfact_exp_dir = os.path.join("results", CHECKPOINT, "analysis", "counterfac-exp", FEATURE_SPACE)
if not os.path.isdir(counterfact_exp_dir):
    os.makedirs(counterfact_exp_dir)



# Open file to append errors
if os.path.exists(os.path.join(counterfact_exp_dir, "err_report.txt")):
    os.remove(os.path.join(counterfact_exp_dir, "err_report.txt"))

err_report = open(os.path.join(counterfact_exp_dir, "err_report.txt"), "at")
best_cases_report = open(os.path.join(counterfact_exp_dir, "best_cases_report.txt"), "at")

# Iterate through the rows of the image_retrieval_df
for index, row in image_retrieval_df.iterrows():

    # Get query image and label
    query_img_fname = row["Image"]
    query_img_label = row["Image Label"]


    # For the CUB2002011 database, we have to apply a special processing
    if DATASET == "CUB2002011":
        
        # Get simplified name
        query_img_ = query_img_fname.split('_')
        query_img_ = query_img_[0:-2]
        query_img_ = '_'.join(query_img_)
        query_img_ = query_img_.lower()

        # Obtain correspondent directory of this image
        query_img_dir = test_map_dirs_dict[query_img_]

        # Get image name
        query_img = Image.open(os.path.join(test_data_path, query_img_dir, query_img_fname)).convert("RGB")

    else:
        query_img = Image.open(os.path.join(test_data_path, query_img_fname.split('.')[0], query_img_fname)).convert("RGB")
    
    
    # Open the query image
    query_img = query_img.resize((224, 224))
    query_img = np.array(query_img)



    # Get counterfactual (not all images of all models have counterfactuals)
    try:
        counterfact_img_fname = row["Nearest Counterfactual"]
        counterfact_label = row["Nearest Counterfactual Label"]


        # For the CUB2002011 database, we have to apply a special processing
        if DATASET == "CUB2002011":
        
            # Get simplified name
            counterfact_img_fname_ = counterfact_img_fname.split('_')
            counterfact_img_fname_ = counterfact_img_fname_[0:-2]
            counterfact_img_fname_ = '_'.join(counterfact_img_fname_)
            counterfact_img_fname_ = counterfact_img_fname_.lower()

            # Obtain correspondent directory of this image
            counterfact_img_dir = test_map_dirs_dict[counterfact_img_fname_]

            # Get counterfactual image path
            counterfact_img_path = os.path.join(test_data_path, counterfact_img_dir, counterfact_img_fname) if os.path.exists(os.path.join(test_data_path, counterfact_img_dir, counterfact_img_fname)) else os.path.join(train_data_path, counterfact_img_dir, counterfact_img_fname)

        else:
            counterfact_img_path = os.path.join(test_data_path, counterfact_img_fname.split('.')[0], counterfact_img_fname) if os.path.exists(os.path.join(test_data_path, counterfact_img_fname.split('.')[0], counterfact_img_fname)) else os.path.join(train_data_path, counterfact_img_fname.split('.')[0], counterfact_img_fname)
        
        
        # Open counterfactual image
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
        # Get the path according to the dataset
        if DATASET == "CUB2002011":
            save_path = os.path.join(counterfact_exp_dir, query_img_dir, query_img_fname.split('.')[0])
        else:
            save_path = os.path.join(counterfact_exp_dir, query_img_fname.split('.')[0])
        
        
        # Create this directory
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        
        # Save results
        # plt.show(block=True)
        plt.savefig(fname=os.path.join(save_path, "query_vs_cntf_global.png"))
        plt.close()



        # Read the query image prototypes
        if DATASET == "CUB2002011":
            query_img_prototypes_path = os.path.join("results", CHECKPOINT, "analysis", "local", query_img_dir, query_img_fname.split('.')[0], "most_activated_prototypes")
        else:
            query_img_prototypes_path = os.path.join("results", CHECKPOINT, "analysis", "local", query_img_fname.split('.')[0], query_img_fname.split('.')[0], "most_activated_prototypes")
        
        
        # Get the query image prototypes
        query_img_prototypes = list()
        
        # In the Deformable ProtoPNet, the prototype retrieval has a different naming style (examples below)
        # top-3_activated_prototype.png
        # top-5_activated_prototype_5-with_box.png
        for i in range(10):

            # Check the basic naming style (inherited from ProtoPNet)
            proto_path = os.path.join(query_img_prototypes_path, f"top-{i+1}_activated_prototype_with_box.png")
            if os.path.exists(proto_path):
                query_img_prototypes.append(proto_path)
            
            # The other option is to check other naming styles
            else:
                for j in range(N_CLASSES*10):
                    proto_path = os.path.join(query_img_prototypes_path, f"top-{i+1}_activated_prototype_{j}-with_box.png")

                    if os.path.exists(proto_path):
                        break

                query_img_prototypes.append(proto_path)


        # Open all the query image prototypes
        query_img_prototypes = [np.array(Image.open(i).convert("RGB")) for i in query_img_prototypes]
        
        # Get the query image prototypes class identities
        # print("Why?")
        cls_ids_query_img = proto_stats_df[proto_stats_df["Image Filename"]==query_img_fname]["Top-10 Prototypes Class Identities"].values[0]
        cls_ids_query_img = cls_ids_query_img.split('[')[1]
        cls_ids_query_img = cls_ids_query_img.split(']')[0]
        cls_ids_query_img = [i for i in cls_ids_query_img.split(',')]
        # print(f'Class Identities of the Prototypes Activated by Query Image: {cls_ids_query_img}')
        
        
        

        # Read the counterfactual image prototypes
        if DATASET == "CUB2002011":
            counterfact_img_prototypes_path = os.path.join("results", CHECKPOINT, "analysis", "local", counterfact_img_dir, counterfact_img_fname.split('.')[0], "most_activated_prototypes")
        else:
            counterfact_img_prototypes_path = os.path.join("results", CHECKPOINT, "analysis", "local", counterfact_img_fname.split('.')[0], counterfact_img_fname.split('.')[0], "most_activated_prototypes")


        # Get the counterfactual image prototypes
        counterfact_img_prototypes = list()
        
        # In the Deformable ProtoPNet, the prototype retrieval has a different naming style (examples below)
        # top-3_activated_prototype.png
        # top-5_activated_prototype_5-with_box.png
        for i in range(10):

            # Check the basic naming style (inherited from ProtoPNet)
            proto_path = os.path.join(counterfact_img_prototypes_path, f"top-{i+1}_activated_prototype_with_box.png")
            if os.path.exists(proto_path):
                counterfact_img_prototypes.append(proto_path)
            
            # The other option is to check other naming styles
            else:
                for j in range(N_CLASSES*10):
                    proto_path = os.path.join(counterfact_img_prototypes_path, f"top-{i+1}_activated_prototype_{j}-with_box.png")

                    if os.path.exists(proto_path):
                        break

                counterfact_img_prototypes.append(proto_path)
        

        # Open all the counterfactual image prototypes
        counterfact_img_prototypes = [np.array(Image.open(i).convert("RGB")) for i in counterfact_img_prototypes]
        
        # Get the counterfactual image prototypes class identities
        cls_ids_counterfact_img = proto_stats_df[proto_stats_df["Image Filename"]==counterfact_img_fname]["Top-10 Prototypes Class Identities"].values[0]
        cls_ids_counterfact_img = cls_ids_counterfact_img.split('[')[1]
        cls_ids_counterfact_img = cls_ids_counterfact_img.split(']')[0]
        cls_ids_counterfact_img = [i for i in cls_ids_counterfact_img.split(',')]
        # print(f'Class Identities of the Prototypes Activated by Counterfactual Image: {cls_ids_counterfact_img}')



        # Plot the prototypes
        for l in range(10):
            
            # Create subfigures 
            fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

            # Plot the query image prototypes in the first column
            ax1.set_title(f"{cls_ids_query_img[l]}")
            ax1.axis("off")
            ax1.imshow(query_img_prototypes[l])


            # Plot the counterfactual image prototypes in the second column
            ax2.set_title(f"{cls_ids_counterfact_img[l]}")
            ax2.axis("off")
            ax2.imshow(counterfact_img_prototypes[l])
        

            # Save figure
            plt.savefig(fname=os.path.join(save_path, f"query_vs_cntf_proto_{l+1}.png"))
            plt.close()


            # FIXME: Extra
            if "query_vs_cntf_proto.png" in os.listdir(save_path):
                os.remove(os.path.join(save_path, "query_vs_cntf_proto.png"))

    except:
        err_report.write(f"Proper counterfactual not available for {query_img_fname}\n")



# Close error report
err_report.close()

print("Finished.")
