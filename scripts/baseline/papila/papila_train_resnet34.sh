#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=pla_r34                 # Job name
#SBATCH -o pla_r34.out                  # STDOUT
#SBATCH -e pla_r34.err                  # STDERR



echo "PAPILA | Started | Training"

echo "Baseline | ResNet34"
python src/baseline/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'

echo "PAPILA | Finished | Training"