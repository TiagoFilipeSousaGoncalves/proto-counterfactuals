#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=ph2_d121                # Job name
#SBATCH -o ph2_d121.out                  # STDOUT
#SBATCH -e ph2_d121.err                  # STDERR



echo "Started | PH2 | Training"
echo "PH2 | Deformable-ProtoPNet DenseNet121"
python src/deformable-protopnet/models_train.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --batchsize 64 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 0 \
 --gpu_id 0 \
 --folds 0 1 2 3 4 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'
 # --timestamp ''

echo "PH2 | Finished | Training"
