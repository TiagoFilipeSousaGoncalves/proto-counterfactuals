# Imports
import os
import argparse
import numpy as np
import pandas as pd



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--checkpoint', type=str, default="data", help="Path to the model checkpoint.")



# Parse the arguments
args = parser.parse_args()



# Get the path of the CSV that contains the analysis
CHECKPOINT = args.checkpoint



# Open the .CSV file
proto_stats_df = pd.read_csv(filepath_or_buffer=os.path.join("results", CHECKPOINT, "analysis", "counterfac-inf", "inference_results.csv"), sep=",", header=0)
# print(proto_stats_df.head())


# Get rows where ground-truth is equal to the predicted label
proto_stats_pr_df = proto_stats_df.copy()[["Image Index", "Ground-Truth Label", "Predicted Label", "Predicted Scores", "Counterfactuals"]][proto_stats_df["Ground-Truth Label"]==proto_stats_df["Predicted Label"]]
# print(proto_stats_pr_df.head())


# Reset index so that indices match the number of rows
proto_stats_pr_df = proto_stats_pr_df.reset_index()



# Generate a matrix to get the frequencies
gt_labels = np.unique(proto_stats_df.copy()["Ground-Truth Label"].values)
print(f"Range of gt_labels: {len(gt_labels)}")
cf_freqs = np.zeros(shape=(len(gt_labels), len(gt_labels)))


# Iterate through rows
for index, row in proto_stats_pr_df.iterrows():

    # Get predicted label
    label = int(row["Ground-Truth Label"])

    # Get proposed counterfactual
    cfact = int(row["Counterfactuals"])

    # Add this to the frequency matrix
    cf_freqs[label, cfact] += 1



# print(proto_stats_pr_df.head())


# Open a file to save a small report w/ .TXT extension
report = open(os.path.join("results", CHECKPOINT, "analysis", "counterfac-inf", "inf_stats.txt"), "at")



# Iterate through frequencies
for idx, row in (cf_freqs):

    # Label
    print(f"Label: {idx}")
    report.write(f"Label: {idx}\n")

    # Counterfactuals
    cfs = np.nonzero(row)
    print(f"Possible Counterfactuals: {cfs}")
    report.write(f"Possible Counterfactuals: {cfs}\n")

    # Add line
    print("\n")
    report.write("\n")



# Close report
report.close()
