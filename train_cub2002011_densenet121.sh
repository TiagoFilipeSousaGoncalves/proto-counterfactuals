#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                   # Partition
#SBATCH --job-name=cub_d121                 # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR



echo "CUB2002011 | START"

# ProtoPNet
# echo "ProtoPNet | DenseNet121"
# python code/models_train.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 32 --num_workers 3 --gpu_id 0

# Deformable-ProtoPNet
echo "Deformable-ProtoPNet | DenseNet121"
python code/protopnet_deform/models_train.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 32 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0

echo "CUB2002011 | FINISHED"
