#!/bin/bash
#
#SBATCH -p rtx2080ti_11GB
#SBATCH --job-name=tr_stan_dn161          # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "STANFORDCARS DenseNet161"

python code/models_train.py --dataset STANFORDCARS --base_architecture densenet161 --batchsize 16 --num_workers 0 --gpu_id 0

echo "Finished"
