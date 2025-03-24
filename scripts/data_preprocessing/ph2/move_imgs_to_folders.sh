#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=ph2-p-mvi                 # Job name
#SBATCH -o ph2-p-mvi.out                  # STDOUT
#SBATCH -e ph2-p-mvi.err                  # STDERR

echo 'Started move-images-to-folders of PH2.'

python src/data_preprocessing/ph2/move_imgs_to_folders.py --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database'

echo 'Finished move-images-to-folders of PH2.'