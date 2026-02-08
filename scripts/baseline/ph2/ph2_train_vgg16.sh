#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=ph2_v16                 # Job name
#SBATCH -o ph2_v16.out                  # STDOUT
#SBATCH -e ph2_v16.err                  # STDERR



echo "PH2 | Started | Training"

echo "Baseline | VGG16"
python src/baseline/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'

echo "PH2 | Finished | Training"