#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=ph2_test                 # Job name
#SBATCH -o ph2_test.out                  # STDOUT
#SBATCH -e ph2_test.err                  # STDERR



echo "PH2 | Started | Testing"

echo "PH2 | ProtoPNet | DenseNet121"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/ph2/protopnet/densenet121/FILL_IN'

echo "PH2 | ProtoPNet | ResNet34"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/ph2/protopnet/resnet34/FILL_IN'

echo "PH2 | ProtoPNet | VGG16"
python src/protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --batchsize 16 \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/ph2/protopnet/vgg16/FILL_IN'

echo "PH2 | Finished | Testing"
