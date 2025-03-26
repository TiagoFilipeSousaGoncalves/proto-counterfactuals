#!/bin/bash
#
#SBATCH -p gpu_min11gb                 # Partition
#SBATCH --qos=gpu_min11gb                   # QOS
#SBATCH --job-name=pla_r34                  # Job name
#SBATCH -o pla_r34.out                  # STDOUT
#SBATCH -e pla_r34.err                  # STDERR



echo "Started | PAPILA | Training"
echo "PAPILA | Deformable-ProtoPNet ResNet34"

python src/deformable-protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0

echo "Finished | PAPILA | Training"