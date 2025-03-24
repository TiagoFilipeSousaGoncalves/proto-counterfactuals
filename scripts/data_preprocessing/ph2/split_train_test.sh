#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=pla-p-tts                 # Job name
#SBATCH -o ph2_tts.out                  # STDOUT
#SBATCH -e ph2_tts.err                  # STDERR

echo 'Started split train-val-test of PH2.'

python src/data_preprocessing/ph2/split_train_test.py --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database'

echo 'Finished split train-val-test of PH2.'