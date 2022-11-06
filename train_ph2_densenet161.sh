#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition
#SBATCH --job-name=ph2_d161               # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "PH2 DenseNet161"

python code/models_train.py --dataset PH2 --base_architecture densenet161 --batchsize 16 --num_workers 0 --gpu_id 0

echo "Finished"
