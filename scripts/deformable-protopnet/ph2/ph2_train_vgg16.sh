#!/bin/bash
#
#SBATCH -p gpu_min80gb                 # Partition
#SBATCH --qos=gpu_min80gb                   # QOS
#SBATCH --job-name=ph2_v16                  # Job name
#SBATCH -o ph2_v16.out                  # STDOUT
#SBATCH -e ph2_v16.err                  # STDERR



echo "Started | PH2 | Training"
echo "PH2 | Deformable-ProtoPNet VGG16"

python src/deformable-protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --batchsize 64 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0

echo "Finished | PH2 | Training"