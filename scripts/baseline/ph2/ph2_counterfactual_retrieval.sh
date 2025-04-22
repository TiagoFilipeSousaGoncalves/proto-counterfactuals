#!/bin/bash
#SBATCH -p gpu_min8gb                      # Partition
#SBATCH --qos=gpu_min8gb                   # QOS
#SBATCH --job-name=ph2_cntret              # Job name
#SBATCH -o ph2_cntret.out                  # STDOUT
#SBATCH -e ph2_cntret.err                  # STDERR



echo "PH2 | Started | Counterfactual Retrieval"
echo "PH2 | Baseline | Convolution Feature Space"
python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/densenet121/2025-03-24_11-14-24 \
 --feature_space conv_features

python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/resnet34/2025-03-24_11-59-55 \
 --feature_space conv_features

python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/vgg16/2025-03-24_14-36-33 \
 --feature_space conv_features
echo "PH2 | Finished | Counterfactual Retrieval"