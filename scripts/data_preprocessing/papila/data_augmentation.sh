#!/bin/bash
#SBATCH --partition=fct                 
#SBATCH --qos=gpu082112025    
#SBATCH --mem=12288M  
#SBATCH --job-name=pla-p-crp                 # Job name
#SBATCH -o pla-p-da.out                  # STDOUT
#SBATCH -e pla-p-da.err                  # STDERR

echo 'Started data-augmentation of PAPILA.'

python src/data_preprocessing/data_augmentation.py \
 --dataset 'papila' \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images' \
 --folds 4

echo 'Finished data-augmentation of PAPILA.'