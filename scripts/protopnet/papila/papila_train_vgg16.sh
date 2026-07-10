#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=pla_v16                # Job name
#SBATCH -o pla_v16.out                  # STDOUT
#SBATCH -e pla_v16.err                  # STDERR



echo "PAPILA | Started | Training"

echo "ProtoPNet | VGG16"
python src/protopnet/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --batchsize 32 \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'
 # --timestamp ''

echo "PAPILA | Finished | Training"
