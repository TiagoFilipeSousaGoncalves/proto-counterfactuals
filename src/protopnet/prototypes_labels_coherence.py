# Imports
import os
import argparse
import numpy as np
import pickle
from itertools import combinations
from scipy.stats import wasserstein_distance
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["cub2002011", "papila", "ph2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")
    parser.add_argument('--result_dir_list', action="append", type=str, required=True, help="Paths to the models result directories.")
    parser.add_argument("--coherence_metric", type=str, required=True, choices=["fleiss_kappa", "earth_movers_distance"])
    args = parser.parse_args()


    # Constants
    DATASET = args.dataset
    RESULT_DIR_LIST = args.result_dir_list
    COHERENCE_METRIC = args.coherence_metric



    # Get the number of classes for the computation of the coherence metric
    if DATASET == "cub2002011": 
        N_CLASSES = 200
    elif DATASET == "papila":
        N_CLASSES = 3
    elif DATASET == "ph2":
        N_CLASSES = 3
    elif DATASET == "STANFORDCARS":
        pass



    # Create a list of multiple checkpoints' stats'
    proto_stats_list = list()

    # Populate the list
    for results_dir in RESULT_DIR_LIST:
        
        # Open the .CSV file
        proto_stats_pkl_fpath = os.path.join(results_dir, "analysis", "local", "analysis.pkl")
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

        # Append to the list
        proto_stats_list.append(proto_stats_pr)



    # Create a dictionary to append this results
    label_coherence_dict = dict()
    coherence_values = list()



    # Go through the list of proto stats df
    for proto_stats_pr_ in proto_stats_list:

        # Iterate through rows of the current df
        for j in range(len(proto_stats_pr_["Image Filename"])):

            # Get image filename and create an entrance in the label_coherence_dict (if not available)
            if proto_stats_pr_["Image Filename"][j] not in label_coherence_dict.keys():
                label_coherence_dict[proto_stats_pr_["Image Filename"][j]] = dict()
                label_coherence_dict[proto_stats_pr_["Image Filename"][j]]["Ground-Truth Label"] = 0
                label_coherence_dict[proto_stats_pr_["Image Filename"][j]]["Top-10 Prototypes Models"] = list()
                label_coherence_dict[proto_stats_pr_["Image Filename"][j]]["Prototype Label Coherence"] = 0


            # Get label and add it to our dictionary of results
            label = proto_stats_pr_["Ground-Truth Label"][j]
            label_coherence_dict[proto_stats_pr_["Image Filename"][j]]["Ground-Truth Label"] = label


            # Get the cls identity of top-k most activated prototypes
            top_k_proto = proto_stats_pr_["Top-10 Prototypes Class Identities"][j]


            # Append this to the dictionary of results
            label_coherence_dict[proto_stats_pr_["Image Filename"][j]]["Top-10 Prototypes Models"].append(top_k_proto)



    # Iterate through the dictionary of results
    for image_filename in label_coherence_dict.keys():

        # Compute only for the cases where we have all the n models
        if len(label_coherence_dict[image_filename]["Top-10 Prototypes Models"]) == len(RESULT_DIR_LIST):
            
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