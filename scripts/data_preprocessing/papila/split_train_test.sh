#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=pla-p-tts                 # Job name
#SBATCH -o papila_tts.out                  # STDOUT
#SBATCH -e papila_tts.err                  # STDERR

echo 'Started split train-val-test of PAPILA.'

python src/data_preprocessing/papila/split_train_test.py --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images'

echo 'Finished split train-val-test of PAPILA.'