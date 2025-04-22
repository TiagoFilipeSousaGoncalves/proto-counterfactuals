#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                  # QOS
#SBATCH --job-name=cub_cntret              # Job name
#SBATCH -o cub_cntret.out                  # STDOUT
#SBATCH -e cub_cntret.err                  # STDERR



echo "CUB2002011 | Started | Counterfactual Retrieval"
echo "CUB2002011 | Baseline | Convolution Feature Space"
python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/densenet121/2025-03-12_01-24-26 \
 --feature_space conv_features

python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/resnet34/2025-03-12_09-25-24 \
 --feature_space conv_features

python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/vgg16/2025-03-14_00-14-55 \
 --feature_space conv_features
echo "CUB2002011 | Finished | Counterfactual Retrieval"