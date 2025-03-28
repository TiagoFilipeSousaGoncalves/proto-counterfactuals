#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=cub_test                 # Job name
#SBATCH -o cub_test.out                  # STDOUT
#SBATCH -e cub_test.err                  # STDERR



echo "CUB2002011 | Started | Testing"

echo "CUB2002011 | Baseline"
python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/densenet121/2025-03-12_01-24-26

python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/resnet34/2025-03-12_09-25-24

python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/vgg16/2025-03-14_00-14-55

echo "CUB2002011 | Finished | Testing"