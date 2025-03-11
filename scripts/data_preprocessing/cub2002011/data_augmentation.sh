#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=cub-p-crp                 # Job name
#SBATCH -o cub-p-da.out                  # STDOUT
#SBATCH -e cub-p-da.err                  # STDERR

echo 'Started data-augmentation of CUB2002011.'

python src/data_preprocessing/data_augmentation.py --dataset 'cub2002011' --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset'

echo 'Finished data-augmentation of CUB2002011.'