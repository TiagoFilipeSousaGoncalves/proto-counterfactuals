#!/bin/bash
#
#SBATCH -p rtx2080ti_11GB                   # Partition
#SBATCH --job-name=cub_v16                  # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR


echo "CUB2002011 VGG16"


# ProtoPNet
# echo "ProtoPNet | VGG16"
# python code/models_train.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 64 --num_workers 0 --gpu_id 0

# Deformable-ProtoPNet
echo "Deformable-ProtoPNet | VGG16"
python code/protopnet_deform/models_train.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 32 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0

echo "CUB2002011 | FINISHED"
