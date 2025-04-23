#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                  # QOS
#SBATCH --job-name=pla_cntret              # Job name
#SBATCH -o pla_cntret.out                  # STDOUT
#SBATCH -e pla_cntret.err                  # STDERR



echo "PAPILA | Started | Counterfactual Retrieval"

echo "PAPILA | ProtoPNet | Convolution Feature Space"
python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/' \
 --feature_space conv_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/' \
 --feature_space conv_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24/' \
 --feature_space conv_features


echo "PAPILA | ProtoPNet | Prototype Feature Space"
python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/' \
 --feature_space proto_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/' \
 --feature_space proto_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24/' \
 --feature_space proto_features

echo "CUB2002011 | Finished | Counterfactual Retrieval"