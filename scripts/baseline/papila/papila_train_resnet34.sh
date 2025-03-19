#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                       # QOS
#SBATCH --job-name=pla_r34                 # Job name
#SBATCH -o pla_r34.out                  # STDOUT
#SBATCH -e pla_r34.err                  # STDERR



echo "PAPILA | Started | Training"

echo "Baseline | ResNet34"
python src/baseline/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0

echo "PAPILA | Finished | Training"