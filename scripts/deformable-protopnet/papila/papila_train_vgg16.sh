#!/bin/bash
#
#SBATCH -p gpu_min80gb                 # Partition
#SBATCH --qos=gpu_min80gb                   # QOS
#SBATCH --job-name=pla_v16                  # Job name
#SBATCH -o pla_v16.out                  # STDOUT
#SBATCH -e pla_v16.err                  # STDERR



echo "Started | PAPILA | Training"
echo "PAPILA | Deformable-ProtoPNet VGG16"

python src/deformable-protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --batchsize 64 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0

echo "Finished | PAPILA | Training"