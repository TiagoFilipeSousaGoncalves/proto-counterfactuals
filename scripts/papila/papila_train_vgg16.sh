#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition
#SBATCH --qos=gtx1080ti                   # QOS
#SBATCH --job-name=pap_v16                # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "PAPILA | Started | Training"

model="baseline"

if [ $model == "baseline" ]
then
    echo "Baseline | VGG16"
    python code/baseline/models_train.py --dataset PAPILA --base_architecture vgg16 --batchsize 16 --num_workers 0 --gpu_id 0
elif [ $model == "ppnet" ]
then
    echo "ProtoPNet | VGG16"
    python code/protopnet/models_train.py --dataset PAPILA --base_architecture vgg16 --batchsize 16 --num_workers 0 --gpu_id 0
elif [ $model == "dppnet" ]
then
    echo "Deformable-ProtoPNet | VGG16"
    python code/protopnet_deform/models_train.py --dataset PAPILA --base_architecture vgg16 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0
else
    echo "Error"
fi

echo "PAPILA | Finished | Training"
