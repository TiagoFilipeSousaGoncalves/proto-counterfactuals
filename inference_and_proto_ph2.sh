#!/bin/bash



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STARTED | PH2 | INFERENCE AND PROTOTYPES"

python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture densenet121 --num_workers 2 --gpu_id 0 --checkpoint ph2/densenet121/2022-11-03_11-29-46/
python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture densenet161 --num_workers 2 --gpu_id 0 --checkpoint ph2/densenet161/2022-11-06_17-22-54/
python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture resnet34 --num_workers 2 --gpu_id 0 --checkpoint ph2/resnet34/2022-11-03_11-29-46/
python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture resnet152 --num_workers 2 --gpu_id 0 --checkpoint ph2/resnet152/2022-11-06_17-23-13/
python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture vgg16 --num_workers 2 --gpu_id 0 --checkpoint ph2/vgg16/2022-11-05_00-13-09/
python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture vgg19 --num_workers 2 --gpu_id 0 --checkpoint ph2/vgg19/2022-11-04_17-04-16/


echo "FINISHED | PH2 | INFERENCE AND PROTOTYPES"
