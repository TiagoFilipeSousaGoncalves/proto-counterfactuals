#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=tr_cub                 # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR


echo "CUB2002011 VGG19"

python code/models_train.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 0 --gpu_id 0

echo "Finished"
