#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition
#SBATCH --job-name=ph2_v19                # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "PH2 VGG19"

python code/models_train.py --dataset PH2 --base_architecture vgg19 --batchsize 32 --num_workers 0 --gpu_id 0

echo "Finished"
