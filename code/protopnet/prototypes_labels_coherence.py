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
parser.add_argument('--append-checkpoints', action="append", type=str, required=True, help="Path to the model checkpoint(s).")



# Parse the arguments
args = parser.parse_args()


# Get the dataset
DATASET = args.dataset

# Get the path of the CSV that contains the analysis
CHECKPOINTS = list()
print(args._get_kwargs())
exit()
CHECKPOINTS = args.checkpoints


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



    # Create a anoter column to count the number of prototypes (out of the most activated) that are related to the class identity
    proto_stats_pr_df["Prototype Label Coherence"] = 0
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
            label_coherence_dict[row["Image Filename"]][row["Top-10 Prototypes Models"]] = list()
            label_coherence_dict[row["Image Filename"]]["Prototype Label Coherence"] = 0


        # Get label and add it to our dictionary of results
        label = row["Ground-Truth Label"]
        label_coherence_dict[row["Image Filename"]]["Ground-Truth Label"] = label


        # Get the cls identity of top-k most activated prototypes
        top_k_proto = row["Top-10 Prototypes Class Identities"]

        # Apply a processing to this string
        top_k_proto = top_k_proto.split('[')[1]
        top_k_proto = top_k_proto.split(']')[0]
        top_k_proto = [i for i in top_k_proto.split(',')]
        # print(top_k_proto)

        # Append this to teh dictionary of results
        label_coherence_dict[row["Top-10 Prototypes Models"]].append(top_k_proto)



# Iterate through the dictionary of results
for image_filename in label_coherence_dict.keys():

    # Get the set of top-10 activated prototypes per image
    protypes_among_models = label_coherence_dict[image_filename]["Top-10 Prototypes Models"]
    protypes_among_models = np.array(protypes_among_models)
    fleiss_kappa_format = aggregate_raters(protypes_among_models)
    fleiss_kappa_value = fleiss_kappa(fleiss_kappa_format)

    # Add this to the dictionary
    label_coherence_dict[image_filename]["Prototype Label Coherence"] = fleiss_kappa_value

    # Add this to the list of values
    coherence_values.append(fleiss_kappa_value)



# Open a file to save a small report w/ .TXT extension
if os.path.exists(os.path.join("results", DATASET, "protopnet", "prototype_label_coherence.txt")):
    os.remove(os.path.join("results", DATASET, "protopnet", "prototype_label_coherence.txt"))

report = open(os.path.join("results", DATASET, "protopnet", "prototype_label_coherence.txt"), "at")



# Get the mean value of coherence
mean_value = np.mean(coherence_values)

# Add this value to a report
report.write(f"Coherence (Fleiss Kappa) among models: {mean_value}\n")

# Close report
report.close()



print("Finished.")
