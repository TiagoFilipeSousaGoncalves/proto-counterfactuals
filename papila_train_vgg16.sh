#!/bin/bash
#
#SBATCH -p teslav100_32GB                 # Partition
#SBATCH --job-name=pap_v16                # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "PAPILA | Started | Training"

model="dppnet"

if [ $model == "ppnet" ]
then
    echo "ProtoPNet | ResNet34"
    python code/protopnet/models_train.py --dataset PAPILA --base_architecture vgg16 --batchsize 64 --num_workers 0 --gpu_id 0
elif [ $model == 'dppnet' ]
then
    echo "Deformable-ProtoPNet | ResNet34"
    python code/protopnet_deform/models_train.py --dataset PAPILA --base_architecture vgg16 --batchsize 64 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0
else
    echo "Error"
fi

echo "PAPILA | Finished | Training"
