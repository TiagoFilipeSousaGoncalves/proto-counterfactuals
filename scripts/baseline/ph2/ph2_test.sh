#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=ph2_test                 # Job name
#SBATCH -o ph2_test.out                  # STDOUT
#SBATCH -e ph2_test.err                  # STDERR




echo "PH2 | Started | Testing"

python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --checkpoint /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/densenet121/2025-03-24_11-14-24

python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --checkpoint /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/resnet34/2025-03-24_11-59-55

python src/baseline/models_test.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --batchsize 16 \
 --num_workers 4 \
 --gpu_id 0 \
 --checkpoint /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/vgg16/2025-03-24_14-36-33

echo "PH2 | Finished | Testing"