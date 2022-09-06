#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



echo "STANFORDCARS VGG19"

python code/models_train.py --dataset STANFORDCARS --base_architecture vgg19 --batchsize 16 --num_workers 3 --gpu_id 0

echo "Finished"