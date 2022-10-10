#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=tr_cub                 # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR


echo "CUB2002011 ResNet34"

python code/models_train.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 32 --num_workers 0 --gpu_id 0

echo "Finished"
