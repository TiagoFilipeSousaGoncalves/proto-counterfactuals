#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition
#SBATCH --qos=gtx1080ti                   # QOS
#SBATCH --job-name=ph2_v16                # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



echo "Started | PH2 | Training"

model="baseline"

if [ $model == "baseline" ]
then
    echo "PH2 | Baseline VGG16"
    python code/baseline/models_train.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --num_workers 0 --gpu_id 0
elif [ $model == "ppnet" ]
then
    echo "PH2 | ProtoPNet VGG16"
    python code/protopnet/models_train.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --num_workers 0 --gpu_id 0
elif [ $model == "dppnet" ]
then
    echo "PH2 | Deformable-ProtoPNet VGG16"
    python code/protopnet_deform/models_train.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0
else
    echo "Error"
fi

echo "Finished | PH2 | Training"
