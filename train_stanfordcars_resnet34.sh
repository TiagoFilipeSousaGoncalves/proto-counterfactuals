#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



echo "STANFORDCARS ResNet34"

python code/models_train.py --dataset STANFORDCARS --base_architecture resnet34 --batchsize 32 --num_workers 3 --gpu_id 0

echo "Finished"
