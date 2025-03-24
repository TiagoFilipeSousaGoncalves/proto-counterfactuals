#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                       # QOS
#SBATCH --job-name=ph2_r34                 # Job name
#SBATCH -o ph2_r34.out                  # STDOUT
#SBATCH -e ph2_r34.err                  # STDERR



echo "PH2 | Started | Training"

echo "ProtoPNet | ResNet34"
python src/protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0

echo "PH2 | Finished | Training"