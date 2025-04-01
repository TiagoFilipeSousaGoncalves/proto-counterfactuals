# Imports
import os
import argparse
import numpy as np
import pickle



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="data", help="Path to the model results directory.")
    args = parser.parse_args()

    # Constants
    RESULTS_DIR = args.results_dir



    # Open the .PKL file
    proto_stats_pkl_path = os.path.join(RESULTS_DIR, "analysis", "counterfac-inf", "inference_results.pkl")
    with open(proto_stats_pkl_path, 'rb') as f: 
        proto_stats_pkl = pickle.load(f)

    # Get rows where ground-truth is equal to the predicted label
    proto_stats_pr_pkl = {
        "Image Index":list(),
        "Ground-Truth Label":list(),
        "Predicted Label":list(),
        "Predicted Scores":list(),
        "Counterfactuals":list()
    }

    # Get raw data
    img_idx = proto_stats_pkl["Image Index"]
    y = proto_stats_pkl["Ground-Truth Label"]
    yh = proto_stats_pkl["Predicted Label"]
    yh_s = proto_stats_pkl["Predicted Scores"]
    cnt = proto_stats_pkl["Counterfactuals"]

    # Build processed data
    for i in range(len(img_idx)):
        if int(y[i]) == int(yh[i]):
            proto_stats_pr_pkl["Image Index"].append(img_idx[i])
            proto_stats_pr_pkl["Ground-Truth Label"].append(y[i])
            proto_stats_pr_pkl["Predicted Label"].append(yh[i])
            proto_stats_pr_pkl["Predicted Scores"].append(yh_s[i])
            proto_stats_pr_pkl["Counterfactuals"].append(cnt[i])


    # Generate a matrix to get the frequencies
    gt_labels = np.unique(proto_stats_pkl["Ground-Truth Label"])
    cf_freqs = np.zeros(shape=(len(gt_labels), len(gt_labels)))


    # Get processed data
    p_img_idx = proto_stats_pr_pkl["Image Index"]
    p_y = proto_stats_pr_pkl["Ground-Truth Label"]
    p_yh = proto_stats_pr_pkl["Predicted Label"]
    p_yh_s = proto_stats_pr_pkl["Predicted Scores"]
    p_cnt = proto_stats_pr_pkl["Counterfactuals"]

    for j in range(len(p_img_idx)):

        # Get predicted label
        label = int(p_y[i])

        # Get proposed counterfactual
        cfact = int(p_cnt[i])

        # Add this to the frequency matrix
        cf_freqs[label, cfact] += 1


    # Open a file to save a small report w/ .TXT extension
    if os.path.exists(os.path.join(RESULTS_DIR, "analysis", "counterfac-inf", "inf_stats.txt")):
        os.remove(os.path.join(RESULTS_DIR, "analysis", "counterfac-inf", "inf_stats.txt"))

    report = open(os.path.join(RESULTS_DIR, "analysis", "counterfac-inf", "inf_stats.txt"), "at")



    # Iterate through frequencies
    for idx, row in enumerate(cf_freqs):

        # Label
        # print(f"Label: {idx}")
        report.write(f"Label: {idx}\n")

        # Counterfactuals
        cfs = np.nonzero(row)
        # print(f"Possible Counterfactuals: {cfs}")
        report.write(f"Possible Counterfactuals: {cfs}\n")

        # Add line
        # print("\n")
        report.write("\n")



    # Close report
    report.close()