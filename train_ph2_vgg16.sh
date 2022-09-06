#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



echo "PH2 VGG16"

python code/models_train.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --num_workers 3 --gpu_id 0

echo "Finished"