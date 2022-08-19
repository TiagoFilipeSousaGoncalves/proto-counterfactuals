#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err


# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
# python code/models_train.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 32 --num_workers 3 --gpu_id 0
# python code/models_train.py --dataset CUB2002011 --base_architecture densenet161 --batchsize 16 --num_workers 3 --gpu_id 0
# python code/models_train.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 32 --num_workers 3 --gpu_id 0
python code/models_train.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 32 --num_workers 3 --gpu_id 0
python code/models_train.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --num_workers 3 --gpu_id 0
python code/models_train.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 3 --gpu_id 0

echo "Finished."
