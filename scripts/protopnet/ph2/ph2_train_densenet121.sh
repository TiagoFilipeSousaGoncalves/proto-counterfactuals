#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                       # QOS
#SBATCH --job-name=ph2_d121                # Job name
#SBATCH -o ph2_d121.out                  # STDOUT
#SBATCH -e ph2_d121.err                  # STDERR



echo "PAPILA | Started | Training"

echo "ProtoPNet | DenseNet121"
python src/protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0

echo "PAPILA | Finished | Training"