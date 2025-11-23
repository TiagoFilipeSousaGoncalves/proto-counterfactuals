#!/bin/bash
#SBATCH --partition=fct                 
#SBATCH --qos=gpu082112025    
#SBATCH --mem=12288M  
#SBATCH --job-name=pla-p-crp                 # Job name
#SBATCH -o pla-p-crp.out                  # STDOUT
#SBATCH -e pla-p-crp.err                  # STDERR

echo 'Started split crop-images of PAPILA.'

python src/data_preprocessing/papila/crop_images.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images'

echo 'Finished split crop-images of PAPILA.'