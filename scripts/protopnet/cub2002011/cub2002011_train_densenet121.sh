#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=cub_d121                 # Job name
#SBATCH -o cub_d121.out           # STDOUT
#SBATCH -e cub_d121.err           # STDERR



echo "Started | CUB2002011 | Training"

echo "CUB200211 | ProtoPNet DenseNet121"
python src/protopnet/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'
 # --timestamp ''

echo "CUB2002011 | FINISHED"
