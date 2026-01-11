#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=pla_d121                 # Job name
#SBATCH -o pla_d121.out                  # STDOUT
#SBATCH -e pla_d121.err                  # STDERR



echo "PAPILA | Started | Training"

echo "Baseline | DenseNet121"
python src/baseline/models_train.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'

echo "PAPILA | Finished | Training"
