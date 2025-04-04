# Imports
import os
import argparse
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
proto_stats_df = pd.read_csv(filepath_or_buffer=os.path.join("results", CHECKPOINT, "analysis", "local", "analysis.csv"), sep=",", header=0)
# print(proto_stats_df.head())


# Get rows where ground-truth is equal to the predicted label
proto_stats_pr_df = proto_stats_df.copy()[["Image Filename", "Ground-Truth Label", "Predicted Label", "Number of Prototypes Connected to the Class Identity", "Top-10 Prototypes Class Identities"]][proto_stats_df["Ground-Truth Label"]==proto_stats_df["Predicted Label"]]
# print(proto_stats_pr_df.head())



# Create a anoter column to count the number of prototypes (out of the most activated) that are related to the class identity
proto_stats_pr_df["Out-of-TopK Identity Activated Prototypes"] = 0
# print(proto_stats_pr_df.head())


# Reset index so that indices match the number of rows
proto_stats_pr_df = proto_stats_pr_df.reset_index()

# Iterate through rows
for index, row in proto_stats_pr_df.iterrows():

    # Get label
    label = row["Ground-Truth Label"]

    # Get the cls identity of top-k most activated prototypes
    top_k_proto = row["Top-10 Prototypes Class Identities"]

    # Apply a processing to this string
    top_k_proto = top_k_proto.split('[')[1]
    top_k_proto = top_k_proto.split(']')[0]
    top_k_proto = [i for i in top_k_proto.split(',')]
    # print(top_k_proto)
    
    # Count the number of prototypes that are equal to image class
    count = 0
    
    for p in top_k_proto:
        if int(p) == int(label):
            count += 1
    

    # Update the dataframe
    proto_stats_pr_df.iloc[[index], [-1]] = count


# print(proto_stats_pr_df.head())


# Open a file to save a small report w/ .TXT extension
if os.path.exists(os.path.join("results", CHECKPOINT, "analysis", "local", "proto_stats.txt")):
    os.remove(os.path.join("results", CHECKPOINT, "analysis", "local", "proto_stats.txt"))

report = open(os.path.join("results", CHECKPOINT, "analysis", "local", "proto_stats.txt"), "at")

# Get mean value of top-k cls-identity prototypes using this model
mean_value = proto_stats_pr_df["Out-of-TopK Identity Activated Prototypes"].mean()
# print(f"Number of prototypes per class identity: {10}")
report.write(f"Number of prototypes per class identity: {10}\n")
# print(f"Average number of class-identity prototypes per correctly classified image: {mean_value}")
report.write(f"Average number of class-identity prototypes per correctly classified image: {mean_value}\n")

# Close report
report.close()



print("Finished.")
