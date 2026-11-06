#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=ph2_test                 # Job name
#SBATCH -o ph2_test.out                  # STDOUT
#SBATCH -e ph2_test.err                  # STDERR



echo "PH2 | Started | Testing"

echo "PH2 | Deformable ProtoPNet | DenseNet121"
python src/deformable-protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2026-05-09_01-04-17'

echo "PH2 | Deformable ProtoPNet | ResNet34"
python src/deformable-protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2026-05-13_07-47-27'

echo "PH2 | Deformable ProtoPNet | VGG16"
python src/deformable-protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2026-05-10_07-15-37'

echo "PH2 | Finished | Testing"
