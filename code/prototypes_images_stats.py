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
proto_stats_df = pd.read_csv(filepath_or_buffer=os.path.join(CHECKPOINT, "analysis", "local", "analysis.csv"), sep=",", header=0)
# print(proto_stats_df.head())


# Get rows where ground-truth is equal to the predicted label
proto_stats_pr_df = proto_stats_df.copy()[proto_stats_df["Ground-Truth Label"]==proto_stats_df["Predicted Label"]]
print(proto_stats_pr_df.head())
