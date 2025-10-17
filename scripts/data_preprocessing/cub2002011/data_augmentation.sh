#!/bin/bash
#SBATCH --partition=fct                 
#SBATCH --qos=gpu082112025    
#SBATCH --mem=12288M       
#SBATCH --job-name=cub-p-crp                 # Job name
#SBATCH -o cub-p-da.out                  # STDOUT
#SBATCH -e cub-p-da.err                  # STDERR

echo 'Started data-augmentation of CUB2002011.'

python src/data_preprocessing/data_augmentation.py \
 --dataset 'cub2002011' \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --folds 0 1 2 3 4

echo 'Finished data-augmentation of CUB2002011.'