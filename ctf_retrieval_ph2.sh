#!/bin/bash



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "Started | PH2 | Counterfactual Retrieval"

python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/ --generate_img_features


echo "Finished | PH2 | Counterfactual Retrieval"
