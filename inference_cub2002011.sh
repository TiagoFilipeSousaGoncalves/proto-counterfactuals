#!/bin/bash
#
#SBATCH -p fast_prototyping                   # Partition        (check w/ $sinfo)
#SBATCH --job-name=test_cub2002011          # Job name
#SBATCH -c 2                                # Number of cores
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR



# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "CUB2002011"

python code/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/densenet121/2022-09-14_06-32-51/
python code/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture densenet161 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/densenet161/2022-09-15_16-14-45/
python code/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/resnet34/2022-09-17_17-03-33/
# python code/models_test.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/resnet152/
# python code/models_test.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/vgg16/
# python code/models_test.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint cub2002011/vgg19/

echo "Finished."