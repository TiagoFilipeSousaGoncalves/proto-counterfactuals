#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "PH2"

python code/models_test.py --dataset PH2 --base_architecture densenet121 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint ph2/densenet121/2022-09-03_13-39-19/
python code/models_test.py --dataset PH2 --base_architecture densenet161 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint ph2/densenet161/2022-09-03_15-48-38/
python code/models_test.py --dataset PH2 --base_architecture resnet34 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint ph2/resnet34/2022-09-03_18-36-08/
python code/models_test.py --dataset PH2 --base_architecture resnet152 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint ph2/resnet152/2022-09-03_20-25-40/
python code/models_test.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint ph2/vgg16/2022-09-03_22-57-58/
python code/models_test.py --dataset PH2 --base_architecture vgg19 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint ph2/vgg19/2022-09-04_00-59-09/

echo "Finished."
