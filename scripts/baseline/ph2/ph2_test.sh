#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=ph2_test                 # Job name
#SBATCH -o ph2_test.out                  # STDOUT
#SBATCH -e ph2_test.err                  # STDERR



echo "PH2 | Started | Testing"
echo "PH2 | Baseline"
python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --batchsize 64 \
 --num_workers 0 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --timestamp "2026-02-08_14-22-26" \

python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --batchsize 64 \
 --num_workers 0 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --timestamp "2026-02-08_20-28-03" \

python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --batchsize 64 \
 --num_workers 0 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --timestamp "2026-02-08_20-29-59" \

echo "PH2 | Finished | Testing"
