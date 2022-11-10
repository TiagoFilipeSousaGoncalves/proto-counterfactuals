#!/bin/bash



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STARTED | PH2 | LOCAL ANALYSIS"

python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture densenet121 --batchsize 32 --num_workers 0 --gpu_id 0 --checkpoint ph2/densenet121/2022-11-03_11-29-46/
python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture densenet161 --batchsize 32 --num_workers 0 --gpu_id 0 --checkpoint ph2/densenet161/2022-11-06_17-22-54/
python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture resnet34 --batchsize 32 --num_workers 0 --gpu_id 0 --checkpoint ph2/resnet34/2022-11-03_11-29-46/
python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture resnet152 --batchsize 32 --num_workers 0 --gpu_id 0 --checkpoint ph2/resnet152/2022-11-06_17-23-13/
python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture vgg16 --batchsize 32 --num_workers 0 --gpu_id 0 --checkpoint ph2/vgg16/2022-11-05_00-13-09/
python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture vgg19 --batchsize 32 --num_workers 0 --gpu_id 0 --checkpoint ph2/vgg19/2022-11-04_17-04-16/

echo "FINISHED | PH2 | LOCAL ANALYSIS"
