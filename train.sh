#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err


# CUB2002011
# python code/models_train.py --dataset CUB2002011 --base_architecture resnet18 --batchsize 16 --num_workers 4 --gpu_id 0
# python code/models_train.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --num_workers 4 --gpu_id 0
# python code/models_train.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 4 --gpu_id 0

# STANFORDCARS
python code/models_train.py --dataset STANFORDCARS --base_architecture resnet18 --batchsize 16 --num_workers 4 --gpu_id 0
python code/models_train.py --dataset STANFORDCARS --base_architecture densenet121 --batchsize 16 --num_workers 4 --gpu_id 0
python code/models_train.py --dataset STANFORDCARS --base_architecture vgg19 --batchsize 16 --num_workers 4 --gpu_id 0
