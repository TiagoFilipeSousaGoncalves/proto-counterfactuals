#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=cub-p-crp                 # Job name
#SBATCH -o cub-p-crp.out                  # STDOUT
#SBATCH -e cub-p-crp.err                  # STDERR

echo 'Started split crop-images of CUB2002011.'

python src/data_preprocessing/cub2002011/crop_images.py --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset'

echo 'Finished split crop-images of CUB2002011.'