# Imports
import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import wasserstein_distance
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Dataset
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PAPILA", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")

# Checkpoint(s)
parser.add_argument('--append-checkpoints', action="append", type=str, required=True, help="Path to the model checkpoint(s).")

# Distance type
parser.add_argument("--coherence_metric", type=str, required=True, choices=["fleiss_kappa", "earth_movers_distance"])


# Parse the arguments
args = parser.parse_args()


# Get the dataset
DATASET = args.dataset

# Get the path of the CSV that contains the analysis
CHECKPOINTS = args.append_checkpoints

# Coherence metric
COHERENCE_METRIC = args.coherence_metric



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
    proto_stats_df = pd.read_csv(filepath_or_buffer=os.path.join("results", checkpoint, "analysis", "local", "analysis.csv"), sep=",", header=0)
    # print(proto_stats_df.head())

    # Get rows where ground-truth is equal to the predicted label
    proto_stats_pr_df = proto_stats_df.copy()[["Image Filename", "Ground-Truth Label", "Predicted Label", "Number of Prototypes Connected to the Class Identity", "Top-10 Prototypes Class Identities"]][proto_stats_df["Ground-Truth Label"]==proto_stats_df["Predicted Label"]]
    # print(proto_stats_pr_df.head())

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
        if row["Image Filename"] not in label_coherence_dict.keys():
            label_coherence_dict[row["Image Filename"]] = dict()
            label_coherence_dict[row["Image Filename"]]["Ground-Truth Label"] = 0
            label_coherence_dict[row["Image Filename"]]["Top-10 Prototypes Models"] = list()
            label_coherence_dict[row["Image Filename"]]["Prototype Label Coherence"] = 0


        # Get label and add it to our dictionary of results
        label = row["Ground-Truth Label"]
        label_coherence_dict[row["Image Filename"]]["Ground-Truth Label"] = label


        # Get the cls identity of top-k most activated prototypes
        top_k_proto = row["Top-10 Prototypes Class Identities"]

        # Apply a processing to this string
        top_k_proto = top_k_proto.split('[')[1]
        top_k_proto = top_k_proto.split(']')[0]
        top_k_proto = [int(i) for i in top_k_proto.split(',')]
        # print(top_k_proto)

        # Append this to teh dictionary of results
        label_coherence_dict[row["Image Filename"]]["Top-10 Prototypes Models"].append(top_k_proto)



# Iterate through the dictionary of results
for image_filename in label_coherence_dict.keys():

    # Compute only for the cases where we have all the n models
    if len(label_coherence_dict[image_filename]["Top-10 Prototypes Models"]) == len(CHECKPOINTS):
        
        # Get the set of top-10 activated prototypes per image
        prototypes_among_models = label_coherence_dict[image_filename]["Top-10 Prototypes Models"]
        prototypes_among_models = np.array(prototypes_among_models)

        
        # Using Fleiss Kappa
        if COHERENCE_METRIC == "fleiss_kappa":
            
            # Transpose the vector so we have the right format to compute the Fleiss Kappa
            prototypes_among_models = np.transpose(prototypes_among_models)

            # Compute the Fleiss Kappa
            fleiss_kappa_arr, categories_arr = aggregate_raters(data=prototypes_among_models, n_cat=N_CLASSES)
            fleiss_kappa_value = fleiss_kappa(table=fleiss_kappa_arr, method='uniform')
            coherence_res = fleiss_kappa_value
        

        # Using Earth Movers Distance
        elif COHERENCE_METRIC == "earth_movers_distance":

            # Get the combinations
            idx_comb = combinations(range(len(prototypes_among_models)), 2)
            idx_comb = list(idx_comb)
            
            # Iterate through these combinations
            wass_distances = list()
            for comb in idx_comb:
                wd = wasserstein_distance(prototypes_among_models[comb[0]], prototypes_among_models[comb[1]])
                wass_distances.append(wd)
            
            # Compute coherence as the mean of these distances
            coherence_res = np.mean(wass_distances)



        # Add this to the dictionary
        label_coherence_dict[image_filename]["Prototype Label Coherence"] = coherence_res

        # Add this to the list of values
        coherence_values.append(coherence_res)



# Open a file to save a small report w/ .TXT extension
if os.path.exists(os.path.join("results", DATASET.lower(), "protopnet", "prototype_label_coherence.txt")):
    os.remove(os.path.join("results", DATASET.lower(), "protopnet", "prototype_label_coherence.txt"))

report = open(os.path.join("results", DATASET.lower(), "protopnet", "prototype_label_coherence.txt"), "at")



# Get the mean value of coherence
mean_value = np.mean(coherence_values)

# Add this value to a report
if COHERENCE_METRIC == "fleiss_kappa":
    report.write(f"Coherence (Fleiss Kappa): {mean_value}\n")
elif COHERENCE_METRIC == "earth_movers_distance":
    report.write(f"Coherence (Earth Movers Distance)): {mean_value}\n")

# Close report
report.close()



print("Finished.")
