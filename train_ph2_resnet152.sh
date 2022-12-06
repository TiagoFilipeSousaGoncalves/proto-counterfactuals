#!/bin/bash
#
#SBATCH -p rtx2080ti_11GB                 # Partition
#SBATCH --job-name=ph2_r152               # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "PH2 | STARTED"

# ProtoPNet
echo "ProtoPNet | ResNet152"
python code/protopnet/models_train.py --dataset PH2 --base_architecture resnet152 --batchsize 16 --num_workers 0 --gpu_id 0

# Deformable-ProtoPNet
# echo "Deformable-ProtoPNet | ResNet152"
# python code/protopnet_deform/models_train.py --dataset PH2 --base_architecture resnet152 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0

echo "PH2 | FINISHED"
