#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                       # QOS
#SBATCH --job-name=pla_d121                 # Job name
#SBATCH -o pla_d121.out                  # STDOUT
#SBATCH -e pla_d121.err                  # STDERR



echo "PAPILA | Started | Training"

echo "Baseline | DenseNet121"
python src/baseline/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0

echo "PAPILA | Finished | Training"