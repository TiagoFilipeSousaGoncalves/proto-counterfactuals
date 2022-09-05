#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "CUB2002011"

python code/models_test.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/densenet121/2022-08-10_11-27-14/
python code/models_test.py --dataset CUB2002011 --base_architecture densenet161 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/densenet161/2022-08-12_11-22-42/
python code/models_test.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/resnet34/2022-08-16_07-33-07/
python code/models_test.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/resnet152/2022-08-19_18-07-45/
python code/models_test.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/vgg16/2022-08-23_18-52-48/
python code/models_test.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/vgg19/2022-08-26_08-12-03/

echo "Finished."
