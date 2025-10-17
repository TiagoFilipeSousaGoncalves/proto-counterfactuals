#!/bin/bash
#SBATCH --partition=fct                 
#SBATCH --qos=gpu082112025    
#SBATCH --mem=12288M  
#SBATCH --job-name=cub-p-crp                 # Job name
#SBATCH -o cub-p-crp.out                  # STDOUT
#SBATCH -e cub-p-crp.err                  # STDERR

echo 'Started split crop-images of CUB2002011.'

python src/data_preprocessing/cub2002011/crop_images.py \
 --data_dir '/users5/cpca082112025/shared/datasets/cub2002011-dataset' \
 --n_folds 5

echo 'Finished split crop-images of CUB2002011.'