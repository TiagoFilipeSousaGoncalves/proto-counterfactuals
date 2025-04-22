#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                  # QOS
#SBATCH --job-name=cub_cntret              # Job name
#SBATCH -o cub_cntret.out                  # STDOUT
#SBATCH -e cub_cntret.err                  # STDERR



echo "CUB2002011 | Started | Counterfactual Retrieval"

echo "CUB2002011 | ProtoPNet | Convolution Feature Space"
python src/protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2025-03-14_11-19-56/' \
 --feature_space conv_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2025-03-17_15-17-17/' \
 --feature_space conv_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2025-03-17_15-17-21/' \
 --feature_space conv_features

echo "CUB2002011 | ProtoPNet | Prototype Feature Space"
python src/protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2025-03-14_11-19-56/' \
 --feature_space proto_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2025-03-17_15-17-17/' \
 --feature_space proto_features

python src/protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2025-03-17_15-17-21/' \
 --feature_space proto_features

echo "CUB2002011 | Finished | Counterfactual Retrieval"