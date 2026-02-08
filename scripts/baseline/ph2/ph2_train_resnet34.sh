#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                       # QOS
#SBATCH --job-name=ph2_r34                 # Job name
#SBATCH -o ph2_r34.out                  # STDOUT
#SBATCH -e ph2_r34.err                  # STDERR



echo "PH2 | Started | Training"

echo "Baseline | ResNet34"
python src/baseline/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'

echo "PH2 | Finished | Training"