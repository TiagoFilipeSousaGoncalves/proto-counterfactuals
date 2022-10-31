#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB
#SBATCH --job-name=tr_stan_rn152          # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "STANFORDCARS ResNet152"

python code/models_train.py --dataset STANFORDCARS --base_architecture resnet152 --batchsize 32 --num_workers 0 --gpu_id 0

echo "Finished"
