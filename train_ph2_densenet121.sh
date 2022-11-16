#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition
#SBATCH --job-name=ph2_d121               # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "PH2 DenseNet121"

# ProtoPNet
# echo "ProtoPNet | DenseNet121"
# python code/models_train.py --dataset PH2 --base_architecture densenet121 --batchsize 64 --num_workers 0 --gpu_id 0

# Deformable-ProtoPNet
echo "Deformable-ProtoPNet | DenseNet121"
python code/protopnet_deform/models_train.py --dataset PH2 --base_architecture densenet121 --batchsize 32 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0

echo "PH2 | FINISHED"
