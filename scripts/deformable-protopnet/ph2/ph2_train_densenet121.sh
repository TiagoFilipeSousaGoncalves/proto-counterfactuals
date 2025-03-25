#!/bin/bash
#
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=ph2_d121                 # Job name
#SBATCH -o ph2_d121.out           # STDOUT
#SBATCH -e ph2_d121.err           # STDERR



echo "Started | CUB2002011 | Training"
echo "CUB200211 | Deformable-ProtoPNet DenseNet121"

python src/deformable-protopnet/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0

echo "CUB2002011 | FINISHED"