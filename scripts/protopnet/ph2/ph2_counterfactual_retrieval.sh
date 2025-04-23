#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                  # QOS
#SBATCH --job-name=ph2_cntret              # Job name
#SBATCH -o ph2_cntret.out                  # STDOUT
#SBATCH -e ph2_cntret.err                  # STDERR



echo "PH2 | Started | Counterfactual Retrieval"

echo "PH2 | ProtoPNet | Convolution Feature Space"
python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/densenet121/2025-03-24_16-19-14/' \
 --feature_space conv_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/resnet34/2025-03-24_18-30-11/' \
 --feature_space conv_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/vgg16/2025-03-24_19-06-46/' \
 --feature_space conv_features


echo "PAPILA | ProtoPNet | Prototype Feature Space"
python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/densenet121/2025-03-24_16-19-14/' \
 --feature_space proto_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/resnet34/2025-03-24_18-30-11/' \
 --feature_space proto_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/vgg16/2025-03-24_19-06-46/' \
 --feature_space proto_features

echo "CUB2002011 | Finished | Counterfactual Retrieval"