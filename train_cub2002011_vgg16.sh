#!/bin/bash
#
#SBATCH -p titanxp_12GB                   # Partition        (check w/ $sinfo)
#SBATCH --job-name=tr_cub                 # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR


echo "CUB2002011 VGG16"

NUM_WORKERS=$(nproc) / 4


python code/models_train.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --num_workers $NUM_WORKERS --gpu_id 0

echo "Finished"
