#!/bin/bash



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STARTED | PH2 | Prototype Inference Stats"

python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/densenet121/2022-11-03_11-29-46/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/densenet161/2022-11-06_17-22-54/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/resnet34/2022-11-03_11-29-46/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/resnet152/2022-11-06_17-23-13/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/vgg16/2022-11-05_00-13-09/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/vgg19/2022-11-04_17-04-16/


echo "FINISHED | PH2 | Prototype Inference Stats"
