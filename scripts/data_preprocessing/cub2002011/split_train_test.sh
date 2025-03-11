#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=cub-p-tts                 # Job name
#SBATCH -o cub2002011_tts.out                  # STDOUT
#SBATCH -e cub2002011_tts.err                  # STDERR

echo 'Started split train-val-test of CUB2002011.'

python src/data_preprocessing/cub2002011/split_train_test.py --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset'

echo 'Finished split train-val-test of CUB2002011.'