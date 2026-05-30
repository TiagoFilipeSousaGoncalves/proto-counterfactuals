#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=cub_r34                # Job name
#SBATCH -o cub_r34.out                  # STDOUT
#SBATCH -e cub_r34.err                  # STDERR

echo "Started | CUB2002011 | Training"
echo "CUB200211 | Deformable-ProtoPNet ResNet34"
python src/deformable-protopnet/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --batchsize 32 \
 --subtractive_margin \
 --using_deform \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'
 # --last_layer_fixed \
 # --timestamp ''

echo "CUB2002011 | Finished | Training"
