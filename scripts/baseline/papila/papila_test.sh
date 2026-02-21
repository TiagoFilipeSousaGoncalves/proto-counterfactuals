#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=pla_test                 # Job name
#SBATCH -o pla_test.out                  # STDOUT
#SBATCH -e pla_test.err                  # STDERR



echo "PAPILA | Started | Testing"

echo "PAPILA | Baseline"
python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --timestamp "2026-02-08_22-29-51" \

python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --timestamp "2026-02-08_20-36-35" \

python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --batchsize 64 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --timestamp "2026-02-08_22-29-51" \

echo "PAPILA | Finished | Testing"
