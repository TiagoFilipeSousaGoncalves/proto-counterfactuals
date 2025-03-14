#!/bin/bash
#
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                       # QOS
#SBATCH --job-name=cub_d121                 # Job name
#SBATCH -o cub_d121.out           # STDOUT
#SBATCH -e cub_d121.err           # STDERR



echo "Started | CUB2002011 | Training"

echo "CUB200211 | ProtoPNet DenseNet121"
python src/protopnet/models_train.py \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0

echo "CUB2002011 | FINISHED"