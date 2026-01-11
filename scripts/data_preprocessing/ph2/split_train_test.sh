#!/bin/bash
#SBATCH --partition=fct
#SBATCH --qos=gpu082112025
#SBATCH --mem=12288M
#SBATCH --job-name=ph2_tts                 # Job name
#SBATCH -o ph2_tts.out                  # STDOUT
#SBATCH -e ph2_tts.err                  # STDERR

echo 'Started split train-val-test of PH2.'

python src/data_preprocessing/ph2/split_train_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/ph2-database' \
 --n_folds 5

echo 'Finished split train-val-test of PH2.'