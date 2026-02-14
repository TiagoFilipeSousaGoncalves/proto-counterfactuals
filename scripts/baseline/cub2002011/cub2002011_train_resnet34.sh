#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=cub_r34
#SBATCH -o cub_r34.out
#SBATCH -e cub_r34.err



echo "Started | CUB2002011 | Training"

echo "CUB2002011 | Baseline ResNet34"
python src/baseline/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 1 2 3 4 \
 --timestamp 2026-02-11_10-51-44 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'

echo "CUB2002011 | FINISHED"
