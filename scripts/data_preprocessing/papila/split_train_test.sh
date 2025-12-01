#!/bin/bash
#SBATCH --partition=fct                 
#SBATCH --qos=gpu082112025    
#SBATCH --mem=12288M  
#SBATCH -o papila_tts.out                  # STDOUT
#SBATCH -e papila_tts.err                  # STDERR

echo 'Started split train-val-test of PAPILA.'

python src/data_preprocessing/papila/split_train_test.py \
 --data_dir '/users5/cpca082112025/shared/datasets/papila-dataset-glaucoma-fundus-images'

echo 'Finished split train-val-test of PAPILA.'