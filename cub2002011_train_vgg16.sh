#!/bin/bash
#
#SBATCH -p titanxp_12GB                     # Partition
#SBATCH --job-name=cub_v16                  # Job name
#SBATCH -o slurm.%N.%j.out                  # STDOUT
#SBATCH -e slurm.%N.%j.err                  # STDERR



echo "Started | CUB2002011 | Training"

model="baseline"

if [ $model == "baseline" ]
then
    echo "CUB200211 | Baseline VGG16"
    python code/baseline/models_train.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 32 --num_workers 0 --gpu_id 0
elif [ $model == "ppnet" ]
then
    echo "CUB200211 | ProtoPNet VGG16"
    python code/protopnet/models_train.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 32 --num_workers 0 --gpu_id 0
elif [ $model == "dppnet" ]
then
    echo "CUB200211 | Deformable-ProtoPNet VGG16"
    python code/protopnet_deform/models_train.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 32 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0
else
    echo "Error"
fi

echo "Finished | CUB2002011 | Training"
