#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=pla-p-crp                 # Job name
#SBATCH -o pla-p-da.out                  # STDOUT
#SBATCH -e pla-p-da.err                  # STDERR

echo 'Started data-augmentation of PAPILA.'

python src/data_preprocessing/data_augmentation.py \
 --dataset 'papila' \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images'

echo 'Finished data-augmentation of PAPILA.'