#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=pla-p-crp                 # Job name
#SBATCH -o pla-p-crp.out                  # STDOUT
#SBATCH -e pla-p-crp.err                  # STDERR

echo 'Started split crop-images of PAPILA.'

python src/data_preprocessing/papila/crop_images.py --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images'

echo 'Finished split crop-images of PAPILA.'