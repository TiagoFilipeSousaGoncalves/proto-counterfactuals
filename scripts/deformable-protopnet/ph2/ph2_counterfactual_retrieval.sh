#!/bin/bash
#SBATCH -p gpu_min32gb                     # Partition
#SBATCH --qos=gpu_min32gb                  # QOS
#SBATCH --job-name=ph2_cntret              # Job name
#SBATCH -o ph2_cntret.out                  # STDOUT
#SBATCH -e ph2_cntret.err                  # STDERR



echo "PH2 | Started | Counterfactual Retrieval"

echo "PH2 | Deformable ProtoPNet | Convolution Feature Space"
python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2025-03-25_20-10-27/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2025-03-25_20-54-57/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2025-03-25_18-54-27/ \
 --feature_space conv_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed


echo "PH2 | Deformable ProtoPNet | Prototype Feature Space"
python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2025-03-25_20-10-27/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2025-03-25_20-54-57/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

python src/deformable-protopnet/models_counterfactuals_retrieval.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2025-03-25_18-54-27/ \
 --feature_space proto_features \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed

echo "PH2 | Finished | Counterfactual Retrieval"