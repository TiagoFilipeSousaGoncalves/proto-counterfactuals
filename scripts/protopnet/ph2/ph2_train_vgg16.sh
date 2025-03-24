#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                       # QOS
#SBATCH --job-name=ph2_v16                # Job name
#SBATCH -o ph2_v16.out                  # STDOUT
#SBATCH -e ph2_v16.err                  # STDERR



echo "PH2 | Started | Training"

echo "ProtoPNet | VGG16"
python src/protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0

echo "PH2 | Finished | Training"