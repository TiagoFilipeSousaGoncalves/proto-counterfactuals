#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=cub_test                 # Job name
#SBATCH -o cub_test.out                  # STDOUT
#SBATCH -e cub_test.err                  # STDERR



echo "CUB2002011 | Started | Testing"

echo "CUB2002011 | Deformable ProtoPNet | DenseNet121"
python src/deformable-protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/cub2002011/deformable-protopnet/densenet121/2026-06-09_11-24-14'

echo "CUB2002011 | Deformable ProtoPNet | ResNet34"
python src/deformable-protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture resnet34 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/cub2002011/deformable-protopnet/resnet34/2026-06-02_02-09-33'

echo "CUB2002011 | Deformable ProtoPNet | VGG16"
python src/deformable-protopnet/models_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture vgg16 \
 --batchsize 16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0 \
 --results_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results/cub2002011/deformable-protopnet/vgg16/2026-05-31_08-21-06'

echo "CUB2002011 | Finished | Testing"
