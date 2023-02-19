# Imports
import os
import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PAPILA", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")

# Checkpoints
parser.add_argument('--append-checkpoints', action="append", type=str, required=True, help="Path to the model checkpoint(s).")

# Feature space
parser.add_argument("--feature_space", type=str, required=True, choices=["conv_features", "proto_features"], help="Feature space: convolutional features (conv_features), prototype layer features (proto_features).")



# Parse the arguments
args = parser.parse_args()



# Get the dataset
DATASET = args.dataset

# Get the path of the CSV that contains the analysis
CHECKPOINTS = args.append_checkpoints

# Feature space
FEATURE_SPACE = args.feature_space



# Get the number of classes for the computation of the coherence metric
if DATASET == "CUB2002011": 
    N_CLASSES = 200
elif DATASET == "PAPILA":
    N_CLASSES = 3
elif DATASET == "PH2":
    N_CLASSES = 3
elif DATASET == "STANFORDCARS":
    pass



# Create a list of multiple checkpoints' stats'
proto_stats_df_list = list()

# Populate the list
for checkpoint in CHECKPOINTS:
    
    # Open the .CSV file
    proto_stats_df = pd.read_csv(filepath_or_buffer=os.path.join("results", checkpoint, "analysis", "image-retrieval", FEATURE_SPACE, "analysis.csv"), sep=",", header=0)

    # Get rows where ground-truth is equal to the predicted label
    proto_stats_pr_df = proto_stats_df.copy()

    # Reset index so that indices match the number of rows
    proto_stats_pr_df = proto_stats_pr_df.reset_index()

    # Append to the list
    proto_stats_df_list.append(proto_stats_pr_df)



# Create a dictionary to append this results
label_coherence_dict = dict()
coherence_values = list()



# Go through the list of proto stats df
for proto_stats_pr_df in proto_stats_df_list:

    # Iterate through rows of the current df
    for index, row in proto_stats_pr_df.iterrows():

        # Get image filename and create an entrance in the label_coherence_dict (if not available)
        if row["Image"] not in label_coherence_dict.keys():
            label_coherence_dict[row["Image"]] = dict()
            label_coherence_dict[row["Image"]]["Image Label"] = row["Image Label"]
            label_coherence_dict[row["Image"]]["Nearest Counterfactual"] = list()
            label_coherence_dict[row["Image"]]["Nearest Counterfactual Label"] = list()
            label_coherence_dict[row["Image"]]["Counterfactual Label Coherence"] = 0


        # Get nearest counterfactual and nearest counterfactual label
        nearest_counterfactual = row["Nearest Counterfactual"]
        nearest_counterfactual_label = row["Nearest Counterfactual Label"]
        label_coherence_dict[row["Image"]]["Nearest Counterfactual"].append(nearest_counterfactual)
        label_coherence_dict[row["Image"]]["Nearest Counterfactual Label"].append(int(nearest_counterfactual_label))



# Iterate through the dictionary of results
for image_filename in label_coherence_dict.keys():

    # Compute only for the cases where we have all the n models
    if len(label_coherence_dict[image_filename]["Nearest Counterfactual Label"]) == len(CHECKPOINTS):
        
        # Get the set of nearest counterfactual labels
        counterfactual_labels_among_models = label_coherence_dict[image_filename]["Nearest Counterfactual Label"]
        counterfactual_labels_among_models = np.array(counterfactual_labels_among_models, dtype=np.int)

        # Transpose the vector so we have the right format to compute the Fleiss Kappa
        counterfactual_labels_among_models = np.transpose(counterfactual_labels_among_models)
        
        # Compute the fleiss kappa value
        fleiss_kappa_arr, categories_arr = aggregate_raters(data=counterfactual_labels_among_models, n_cat=N_CLASSES)
        fleiss_kappa_value = fleiss_kappa(table=fleiss_kappa_arr, method='uniform')

        # Add this to the dictionary
        label_coherence_dict[image_filename]["Counterfactual Label Coherence"] = fleiss_kappa_value

        # Add this to the list of values
        coherence_values.append(fleiss_kappa_value)



# Open a file to save a small report w/ .TXT extension
if os.path.exists(os.path.join("results", DATASET.lower(), "deformable-protopnet", f"{FEATURE_SPACE.lower()}_cnt_label_coherence.txt")):
    os.remove(os.path.join("results", DATASET.lower(), "deformable-protopnet", f"{FEATURE_SPACE.lower()}_cnt_label_coherence.txt"))

report = open(os.path.join("results", DATASET.lower(), "deformable-protopnet", f"{FEATURE_SPACE.lower()}_cnt_label_coherence.txt"), "at")



# Get the mean value of coherence
mean_value = np.mean(coherence_values)

# Add this value to a report
report.write(f"Coherence (Fleiss Kappa): {mean_value}\n")

# Close report
report.close()



print("Finished.")
