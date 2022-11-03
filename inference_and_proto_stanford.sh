#!/bin/bash



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STARTED | STANFORDCARS | INFERENCE AND PROTOTYPES"

python code/models_inference_and_prototypes.py --dataset STANFORDCARS --base_architecture densenet121 --num_workers 2 --gpu_id 0 --checkpoint --checkpoint stanfordcars/densenet121/2022-10-24_08-55-48/
# python code/models_inference_and_prototypes.py --dataset STANFORDCARS --base_architecture densenet161 --num_workers 2 --gpu_id 0 --checkpoint stanfordcars/densenet161/
python code/models_inference_and_prototypes.py --dataset STANFORDCARS --base_architecture resnet34 --num_workers 2 --gpu_id 0 --checkpoint stanfordcars/resnet34/2022-10-25_14-20-40/
# python code/models_inference_and_prototypes.py --dataset STANFORDCARS --base_architecture resnet152 --num_workers 2 --gpu_id 0 --checkpoint stanfordcars/resnet152/
# python code/models_inference_and_prototypes.py --dataset STANFORDCARS --base_architecture vgg16 --num_workers 2 --gpu_id 0 --checkpoint stanfordcars/vgg16/2022-10-26_10-49-44/
# python code/models_inference_and_prototypes.py --dataset STANFORDCARS --base_architecture vgg19 --num_workers 2 --gpu_id 0 --checkpoint stanfordcars/vgg19/2022-10-17_08-13-40/


echo "FINISHED | STANFORDCARS | INFERENCE AND PROTOTYPES"
