#!/bin/bash
#SBATCH -p gpu_min11gb                   # Partition
#SBATCH --qos=gpu_min11gb                # QOS
#SBATCH --job-name=cub_test              # Job name
#SBATCH -o cub_test.out                  # STDOUT
#SBATCH -e cub_test.err                  # STDERR



echo "CUB2002011 | Started | Testing"
echo "CUB2002011 | Baseline"
python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --batchsize 64 \
 --num_workers 0 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --timestamp "2026-02-08_14-22-26"

python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --batchsize 64 \
 --num_workers 0 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --timestamp "2026-02-11_10-51-44"

python src/baseline/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --batchsize 16 \
 --num_workers 0 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results' \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --timestamp "2026-02-12_16-23-13"

echo "CUB2002011 | Finished | Testing"
