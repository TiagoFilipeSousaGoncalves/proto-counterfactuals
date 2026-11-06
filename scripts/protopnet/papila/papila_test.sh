#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=pla_test                 # Job name
#SBATCH -o pla_test.out                  # STDOUT
#SBATCH -e pla_test.err                  # STDERR



echo "PAPILA | Started | Testing"

echo "PAPILA | ProtoPNet | DenseNet121"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/papila/protopnet/densenet121/FILL_IN'

echo "PAPILA | ProtoPNet | ResNet34"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/papila/protopnet/resnet34/FILL_IN'

echo "PAPILA | ProtoPNet | VGG16"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/papila/protopnet/vgg16/FILL_IN'

echo "PAPILA | Finished | Testing"
