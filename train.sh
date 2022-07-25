#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err

python code/models_train.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 4