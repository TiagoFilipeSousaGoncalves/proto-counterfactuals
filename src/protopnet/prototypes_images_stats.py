# Imports
import os
import argparse
import pickle
import numpy as np



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True, help="Path to the model results directory.")
    args = parser.parse_args()


    # Constants
    RESULTS_DIR = args.results_dir



    # Open the .CSV file
    proto_stats_pkl_fpath = os.path.join(RESULTS_DIR, "analysis", "local", "analysis.pkl")
    with open(proto_stats_pkl_fpath, "rb") as f:
        proto_stats = pickle.load(f)

    # Get cases where ground-truth is equal to the predicted label for a processed pickle file
    proto_stats_pr = {
        "Image Filename":list(),
        "Ground-Truth Label":list(),
        "Predicted Label":list(),
        "Number of Prototypes Connected to the Class Identity":list(),
        "Top-10 Prototypes Class Identities":list()
    }

    img_fnames = proto_stats["Image Filename"]
    gt_labels = proto_stats["Ground-Truth Label"]
    pred_labels = proto_stats["Predicted Label"]
    nr_proto_cli = proto_stats["Number of Prototypes Connected to the Class Identity"]
    topk_proto_cli = proto_stats["Top-10 Prototypes Class Identities"]

    for i in range(len(img_fnames)):
        if int(gt_labels[i]) == int(pred_labels[i]):
            proto_stats_pr["Image Filename"].append(img_fnames[i])
            proto_stats_pr["Ground-Truth Label"].append(gt_labels[i])
            proto_stats_pr["Predicted Label"].append(pred_labels[i])
            proto_stats_pr["Number of Prototypes Connected to the Class Identity"].append(nr_proto_cli[i])
            proto_stats_pr["Top-10 Prototypes Class Identities"].append(topk_proto_cli[i])



    # Create a anoter list to count the number of prototypes (out of the most activated) that are related to the class identity
    proto_stats_pr["Out-of-TopK Identity Activated Prototypes"] = [0 for _ in range(len(proto_stats_pr["Image Filename"]))]




    # Iterate through rows
    for j in range(len(proto_stats_pr["Image Filename"])):

        # Get label
        label = proto_stats_pr["Ground-Truth Label"][j]

        # Get the cls identity of top-k most activated prototypes
        top_k_proto = proto_stats_pr["Top-10 Prototypes Class Identities"][j]
        
        # Count the number of prototypes that are equal to image class
        count = 0
        
        for p in top_k_proto:
            if int(p) == int(label):
                count += 1
        

        # Update the dictionary/list
        proto_stats_pr["Out-of-TopK Identity Activated Prototypes"][j] = count


    # Open a file to save a small report w/ .TXT extension
    if os.path.exists(os.path.join(RESULTS_DIR, "analysis", "local", "proto_stats.txt")):
        os.remove(os.path.join(RESULTS_DIR, "analysis", "local", "proto_stats.txt"))

    report = open(os.path.join(RESULTS_DIR, "analysis", "local", "proto_stats.txt"), "at")

    # Get mean value of top-k cls-identity prototypes using this model
    mean_value = np.mean(proto_stats_pr["Out-of-TopK Identity Activated Prototypes"])
    report.write(f"Number of prototypes per class identity: {10}\n")
    report.write(f"Average number of class-identity prototypes per correctly classified image: {mean_value}\n")

    # Close report
    report.close()