#!/bin/bash
#
#SBATCH -p gpu_min80gb                 # Partition
#SBATCH --qos=gpu_min80gb                   # QOS
#SBATCH --job-name=cub_v16                  # Job name
#SBATCH -o cub_v16.out                  # STDOUT
#SBATCH -e cub_v16.err                  # STDERR



echo "Started | CUB2002011 | Training"
echo "CUB200211 | Deformable-ProtoPNet VGG16"

python src/deformable-protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --batchsize 64 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0

echo "Finished | CUB2002011 | Training"