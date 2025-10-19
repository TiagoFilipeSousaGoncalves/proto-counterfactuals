#!/bin/bash
#SBATCH --partition=gpu                 
#SBATCH --qos=gpu082112025    
#SBATCH --mem=12288M                   
#SBATCH --job-name=cub_d121
#SBATCH -o cub_d121.out
#SBATCH -e cub_d121.err



echo "Started | CUB2002011 | Training"

echo "CUB200211 | Baseline DenseNet121"
python src/baseline/models_train.py \
 --data_dir 'data/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --batchsize 32 \
 --num_workers 4 \
 --gpu_id 0 \
 --folds 0 1 2 3 \
 --output_dir '/users5/cpca082112025/shared/experiments/tgoncalves/proto-counterfactuals/results'

echo "CUB2002011 | FINISHED"