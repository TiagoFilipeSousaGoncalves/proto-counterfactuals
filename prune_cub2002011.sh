#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"

echo "Prunning | CUB2002011 | STARTED"

python code/models_prototype_pruning.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 32 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint cub2002011/densenet121/2022-08-10_11-27-14/ --k 6 --prune_threshold 3
python code/models_prototype_pruning.py --dataset CUB2002011 --base_architecture densenet161 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint cub2002011/densenet161/2022-08-12_11-22-42/ --k 6 --prune_threshold 3
python code/models_prototype_pruning.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 32 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint cub2002011/resnet34/2022-08-16_07-33-07/ --k 6 --prune_threshold 3
# python code/models_prototype_pruning.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 32 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint cub2002011/ --k 6 --prune_threshold 3
# python code/models_prototype_pruning.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint cub2002011/ --k 6 --prune_threshold 3
# python code/models_prototype_pruning.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint cub2002011/ --k 6 --prune_threshold 3

echo "Prunning | CUB2002011 | FINISHED"
