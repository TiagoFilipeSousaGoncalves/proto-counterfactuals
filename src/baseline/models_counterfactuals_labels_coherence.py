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
    parser.add_argument('--result_dir_list', action="append", type=str, required=True, help="Path to the model results directory(ies).")
    parser.add_argument("--feature_space", type=str, required=True, choices=["conv_features"], help="Feature space: convolutional features (conv_features).")
    parser.add_argument("--coherence_metric", type=str, required=True, choices=["fleiss_kappa", "earth_movers_distance"])
    args = parser.parse_args()



    # Constants
    DATASET = args.dataset
    RESULTS_DIR_LIST = args.result_dir_list
    FEATURE_SPACE = args.feature_space
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
    proto_stats_list = list()

    # Populate the list
    for results_dir in RESULTS_DIR_LIST:
        
        # Open the .CSV file
        with open(os.path.join(results_dir, "analysis", "image-retrieval", FEATURE_SPACE, "analysis.pkl"), 'rb') as f:
            proto_stats = pickle.load(f)

        proto_stats_list.append(proto_stats)



    # Create a dictionary to append this results
    label_coherence_dict = dict()
    coherence_values = list()



    # Go through the list of proto stats df
    for proto_stats in proto_stats_list:
        for j in range(len(proto_stats["Image"])):

            # Get image filename and create an entrance in the label_coherence_dict (if not available)
            if proto_stats["Image"][j] not in label_coherence_dict.keys():
                label_coherence_dict[proto_stats["Image"][j]] = dict()
                label_coherence_dict[proto_stats["Image"][j]]["Image Label"] = proto_stats["Image"][j]
                label_coherence_dict[proto_stats["Image"][j]]["Nearest Counterfactual"] = list()
                label_coherence_dict[proto_stats["Image"][j]]["Nearest Counterfactual Label"] = list()
                label_coherence_dict[proto_stats["Image"][j]]["Counterfactual Label Coherence"] = 0


            # Get nearest counterfactual and nearest counterfactual label
            nearest_counterfactual = proto_stats["Nearest Counterfactual"][j]
            nearest_counterfactual_label = proto_stats["Nearest Counterfactual Label"][j]
            label_coherence_dict[proto_stats["Image"][j]]["Nearest Counterfactual"].append(nearest_counterfactual)
            label_coherence_dict[proto_stats["Image"][j]]["Nearest Counterfactual Label"].append(int(nearest_counterfactual_label))



    # Iterate through the dictionary of results
    for image_filename in label_coherence_dict.keys():

        # Compute only for the cases where we have all the n models
        if len(label_coherence_dict[image_filename]["Nearest Counterfactual Label"]) == len(RESULTS_DIR_LIST):
            
            # Get the set of nearest counterfactual labels
            counterfactual_labels_among_models = label_coherence_dict[image_filename]["Nearest Counterfactual Label"]
            counterfactual_labels_among_models = np.array(counterfactual_labels_among_models, dtype=int)

            # Add another dimension so we have a single-column vector
            counterfactual_labels_among_models = np.reshape(counterfactual_labels_among_models, (-1, 1))

            # Transpose the vector so we have the right format to compute the Fleiss Kappa
            counterfactual_labels_among_models = np.transpose(counterfactual_labels_among_models)



            # Coherence Metric is Fleiss Kappa
            if COHERENCE_METRIC == "fleiss_kappa":

                # Compute the fleiss kappa value
                fleiss_kappa_arr, categories_arr = aggregate_raters(data=counterfactual_labels_among_models, n_cat=N_CLASSES)
                fleiss_kappa_value = fleiss_kappa(table=fleiss_kappa_arr, method='uniform')

                # Coherence value is the fleiss_kappa_value
                coherence_res = fleiss_kappa_value


            # Coherence Metric is Earth Movers Distance
            elif COHERENCE_METRIC == "earth_movers_distance":

                # Flatten the array to make it easier to manipulate
                counterfactual_labels_among_models = counterfactual_labels_among_models.flatten()

                # Get the combinations
                idx_comb = combinations(range(len(counterfactual_labels_among_models)), 2)
                idx_comb = list(idx_comb)
                
                # Iterate through these combinations
                wass_distances = list()
                for comb in idx_comb:
                    wd = wasserstein_distance([counterfactual_labels_among_models[comb[0]]], [counterfactual_labels_among_models[comb[1]]])
                    wass_distances.append(wd)
                
                # Compute coherence as the mean of these distances
                coherence_res = np.mean(wass_distances)



            # Add this to the dictionary
            label_coherence_dict[image_filename]["Counterfactual Label Coherence"] = coherence_res

            # Add this to the list of values
            coherence_values.append(coherence_res)



    # Open a file to save a small report w/ .TXT extension
    if os.path.exists(os.path.join("results", DATASET.lower(), "baseline", f"{FEATURE_SPACE.lower()}_cnt_label_coherence.txt")):
        os.remove(os.path.join("results", DATASET.lower(), "baseline", f"{FEATURE_SPACE.lower()}_cnt_label_coherence.txt"))

    report = open(os.path.join("results", DATASET.lower(), "baseline", f"{FEATURE_SPACE.lower()}_cnt_label_coherence.txt"), "at")



    # Get the mean value of coherence
    mean_value = np.mean(coherence_values)

    # Add this value to a report
    if COHERENCE_METRIC == "fleiss_kappa":
        report.write(f"Coherence (Fleiss Kappa): {mean_value}\n")
    elif COHERENCE_METRIC == "earth_movers_distance":
        report.write(f"Coherence (Earth Movers Distance): {mean_value}\n")

    # Close report
    report.close()