# Imports
import os
import argparse
import pandas as pd



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--csv_analysis_path', type=str, default="data", help="Path to the CSV file with the prototype analysis.")



# Parse the arguments
args = parser.parse_args()



# Get the path of the CSV that contains the analysis
CSV_ANALYSIS_PATH = args.csv_analysis_path



# Open the .CSV file
proto_stats_df = pd.read_csv(filepath_or_buffer=CSV_ANALYSIS_PATH, sep=",", header=0)


print(proto_stats_df.head())
