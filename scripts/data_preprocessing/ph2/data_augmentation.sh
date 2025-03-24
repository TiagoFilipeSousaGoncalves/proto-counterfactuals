#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=ph2-p-da                 # Job name
#SBATCH -o ph2-p-da.out                  # STDOUT
#SBATCH -e ph2-p-da.err                  # STDERR

echo 'Started data-augmentation of PH2.'

python src/data_preprocessing/data_augmentation.py \
 --dataset 'ph2' \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database'

echo 'Finished data-augmentation of PH2.'