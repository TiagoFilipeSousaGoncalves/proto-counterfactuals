#!/bin/bash
#
#SBATCH -p teslav100_32GB                   # Partition
#SBATCH --job-name=cub_d121                 # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR



echo "Started | CUB2002011 | Training"

if [ $1 == "ppnet" ]
then
echo "CUB200211 | ProtoPNet DenseNet121"
    python code/protopnet/models_train.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 128 --num_workers 0 --gpu_id 0
elif [ $1 == 'dppnet' ]
then
    echo "CUB200211 | Deformable-ProtoPNet DenseNet121"
    python code/protopnet_deform/models_train.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 128 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0

echo "CUB2002011 | FINISHED"
