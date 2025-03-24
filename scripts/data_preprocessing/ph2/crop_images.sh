#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=ph2-p-crp                 # Job name
#SBATCH -o ph2-p-crp.out                  # STDOUT
#SBATCH -e ph2-p-crp.err                  # STDERR

echo 'Started split crop-images of PAPILA.'

python src/data_preprocessing/ph2/crop_images.py --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database'

echo 'Finished split crop-images of PAPILA.'