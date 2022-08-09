#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err

python code/models_test.py --dataset CUB2002011 --base_architecture resnet18 --batchsize 16 --num_workers 3 --checkpoint cub2002011/resnet18/2022-08-05_18-17-53
python code/models_test.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --num_workers 3 --checkpoint cub2002011/densenet121/2022-08-07_00-34-22
# python code/models_test.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 4 --checkpoint cub2002011/vgg19/2022-07-29_11-11-33
