#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=cub_test                 # Job name
#SBATCH -o cub_test.out                  # STDOUT
#SBATCH -e cub_test.err                  # STDERR



echo "CUB2002011 | Started | Testing"

echo "CUB2002011 | ProtoPNet | DenseNet121"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2026-07-10_23-30-35'

echo "CUB2002011 | ProtoPNet | ResNet34"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2026-07-19_18-26-02'

echo "CUB2002011 | ProtoPNet | VGG16"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2026-07-19_18-27-21'

echo "CUB2002011 | Finished | Testing"
