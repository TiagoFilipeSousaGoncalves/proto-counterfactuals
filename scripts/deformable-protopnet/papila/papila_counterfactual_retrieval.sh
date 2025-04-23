#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                  # QOS
#SBATCH --job-name=pla_cntret              # Job name
#SBATCH -o pla_cntret.out                  # STDOUT
#SBATCH -e pla_cntret.err                  # STDERR



echo "PAPILA | Started | Counterfactual Retrieval"

echo "PAPILA | Deformable ProtoPNet | Convolution Feature Space"
python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/densenet121/2025-03-26_11-31-33/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/resnet34/2025-03-26_12-12-55/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/vgg16/2025-03-26_11-31-32/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed


echo "PAPILA | Deformable ProtoPNet | Prototype Feature Space"
python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/densenet121/2025-03-26_11-31-33/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/resnet34/2025-03-26_12-12-55/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/vgg16/2025-03-26_11-31-32/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

echo "PAPILA | Finished | Counterfactual Retrieval"