#!/bin/bash
#
#SBATCH -p gpu_min32gb                 # Partition
#SBATCH --qos=gpu_min32gb                   # QOS
#SBATCH --job-name=cub_r34                  # Job name
#SBATCH -o cub_r34.out                  # STDOUT
#SBATCH -e cub_r34.err                  # STDERR



echo "Started | CUB2002011 | Training"
echo "CUB200211 | Deformable-ProtoPNet ResNet34"

python src/deformable-protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --batchsize 32 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0

echo "Finished | CUB2002011 | Training"