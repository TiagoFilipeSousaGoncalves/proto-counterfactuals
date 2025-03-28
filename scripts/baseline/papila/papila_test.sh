#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=pla_test                 # Job name
#SBATCH -o pla_test.out                  # STDOUT
#SBATCH -e pla_test.err                  # STDERR



echo "PAPILA | Started | Testing"

echo "PAPILA | Baseline"
python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/densenet121/2025-03-19_22-34-07

python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/resnet34/2025-03-19_23-07-26

python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/vgg16/2025-03-21_16-13-55

echo "PAPILA | Finished | Testing"