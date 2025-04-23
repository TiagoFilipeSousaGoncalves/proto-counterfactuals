#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                  # QOS
#SBATCH --job-name=cub_cntret              # Job name
#SBATCH -o cub_cntret.out                  # STDOUT
#SBATCH -e cub_cntret.err                  # STDERR



echo "CUB2002011 | Started | Counterfactual Retrieval"

echo "CUB2002011 | Deformable ProtoPNet | Convolution Feature Space"
python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/densenet121/2025-03-24_09-39-04/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/resnet34/2025-03-24_23-56-33/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/vgg16/2025-03-24_23-59-23/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

echo "CUB2002011 | Deformable ProtoPNet | Prototype Feature Space"
python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/densenet121/2025-03-24_09-39-04/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/resnet34/2025-03-24_23-56-33/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --dataset cub2002011 \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/vgg16/2025-03-24_23-59-23/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

echo "CUB2002011 | Finished | Counterfactual Retrieval"