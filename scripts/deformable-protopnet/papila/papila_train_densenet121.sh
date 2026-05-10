#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=papila_d121                # Job name
#SBATCH -o papila_d121.out                  # STDOUT
#SBATCH -e papila_d121.err                  # STDERR


echo "Started | PAPILA | Training"
echo "PAPILA | Deformable-ProtoPNet DenseNet121"
python src/deformable-protopnet/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --batchsize 32 \
 --subtractive_margin \
 --using_deform \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'
 # --last_layer_fixed \
 # --timestamp ''

echo "PAPILA | Finished | Training"
