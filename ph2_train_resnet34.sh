#!/bin/bash
#
#SBATCH -p titanxp_12GB                   # Partition
#SBATCH --job-name=ph2_r34                # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "Started | PH2 | Training"

model="dppnet"

if [ $model == "dppnet" ]
then
    echo "PH2 | ProtoPNet ResNet34"
    python code/protopnet/models_train.py --dataset PH2 --base_architecture resnet34 --batchsize 16 --num_workers 0 --gpu_id 0
elif [ $model == 'dppnet' ]
then
    echo "PH2 | Deformable-ProtoPNet ResNet34"
    python code/protopnet_deform/models_train.py --dataset PH2 --base_architecture resnet34 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0
else
    echo "Error"
fi

echo "Finished | PH2 | Training"
