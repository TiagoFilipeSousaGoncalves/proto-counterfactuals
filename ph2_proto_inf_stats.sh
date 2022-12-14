#!/bin/bash



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "Started | PH2 | Prototype Inference Stats"


echo "PH2 | ProtoPNet"
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/
python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/


echo "Finished | PH2 | Prototype Inference Stats"
