#!/bin/bash
#SBATCH -p debug_8gb                     # Partition
#SBATCH --qos=debug_8gb                       # QOS
#SBATCH --job-name=cub-p-tts                 # Job name
#SBATCH -o cub2002011_tts.out                  # STDOUT
#SBATCH -e cub2002011_tts.err                  # STDERR

echo 'Started split train-val-test of CUB2002011.'

python src/data_preprocessing/cub2002011/split_train_test.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset/CUB_200_2011' \
 --seed 42

echo 'Finished split train-val-test of CUB2002011.'