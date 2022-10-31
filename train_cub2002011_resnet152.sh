#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition
#SBATCH --job-name=tr_cub                 # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR


echo "STARTED | CUB2002011 | TRAIN ResNet152"

# NUM_WORKERS=$(nproc)

python code/models_train.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 32 --num_workers 0 --gpu_id 0

echo "FINISHED | CUB2002011 | TRAIN ResNet152"
