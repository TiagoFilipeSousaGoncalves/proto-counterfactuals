#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                  # QOS
#SBATCH --job-name=pla_cntret              # Job name
#SBATCH -o pla_cntret.out                  # STDOUT
#SBATCH -e pla_cntret.err                  # STDERR



echo "PAPILA | Started | Counterfactual Retrieval"
echo "PAPILA | Baseline | Convolution Feature Space"
python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/densenet121/2025-03-19_22-34-07 \
 --feature_space conv_features

python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/resnet34/2025-03-19_23-07-26 \
 --feature_space conv_features

python src/baseline/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/vgg16/2025-03-21_16-13-55 \
 --feature_space conv_features
echo "PAPILA | Finished | Counterfactual Retrieval"