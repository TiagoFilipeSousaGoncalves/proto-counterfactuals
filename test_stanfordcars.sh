#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STANFORDCARS"

python code/models_test.py --dataset STANFORDCARS --base_architecture densenet121 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/densenet121/2022-08-11_16-16-36/
python code/models_test.py --dataset STANFORDCARS --base_architecture densenet161 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/densenet161/2022-08-21_00-52-17/
python code/models_test.py --dataset STANFORDCARS --base_architecture resnet34 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/resnet34/2022-08-14_13-44-35/
python code/models_test.py --dataset STANFORDCARS --base_architecture resnet152 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/resnet152/2022-08-26_10-59-27/
python code/models_test.py --dataset STANFORDCARS --base_architecture vgg16 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/vgg16/2022-09-02_13-24-39/
python code/models_test.py --dataset STANFORDCARS --base_architecture vgg19 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/vgg19/2022-09-04_11-05-28/

echo "Finished."
