#!/bin/bash
#
#SBATCH -p titanxp_12GB                     # Partition
#SBATCH --job-name=cub_v19                  # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR



echo "CUB2002011 | START"

# ProtoPNet
echo "ProtoPNet | VGG19"
python code/protopnet/models_train.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 0 --gpu_id 0

# Deformable-ProtoPNet
# echo "Deformable-ProtoPNet | VGG19"
# python code/protopnet_deform/models_train.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0

echo "CUB2002011 | FINISHED"
