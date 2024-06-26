#!/bin/bash
#
#SBATCH -p titanxp_12GB
#SBATCH --job-name=tr_stan_vgg16          # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "STANFORDCARS VGG16"

python code/models_train.py --dataset STANFORDCARS --base_architecture vgg16 --batchsize 64 --num_workers 0 --gpu_id 0

echo "Finished"
