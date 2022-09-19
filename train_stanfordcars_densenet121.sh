#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=tr_stan_dn121          # Job name
#SBATCH -c 3                              # Number of cores
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "STANFORDCARS DenseNet121"

python code/models_train.py --dataset STANFORDCARS --base_architecture densenet121 --batchsize 64 --num_workers 0 --gpu_id 0

echo "Finished"
