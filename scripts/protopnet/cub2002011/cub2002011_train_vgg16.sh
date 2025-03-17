#!/bin/bash
#
#SBATCH -p gpu_min32gb                 # Partition
#SBATCH --qos=gpu_min32gb                   # QOS
#SBATCH --job-name=cub_v16                  # Job name
#SBATCH -o cub_v16.out                  # STDOUT
#SBATCH -e cub_v16.err                  # STDERR



echo "Started | CUB2002011 | Training"

echo "CUB200211 | ProtoPNet VGG16"
python src/protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0

echo "Finished | CUB2002011 | Training"