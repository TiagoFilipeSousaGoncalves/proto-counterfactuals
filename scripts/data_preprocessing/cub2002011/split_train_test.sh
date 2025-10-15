#!/bin/bash
#SBATCH --partition=fct                 
#SBATCH --qos=gpu082112025    
#SBATCH --mem=12288M                   
#SBATCH --job-name=cub-p-tts                 # Job name
#SBATCH -o cub2002011_tts.out                  # STDOUT
#SBATCH -e cub2002011_tts.err                  # STDERR

echo 'Started split train-val-test of CUB2002011.'

python src/data_preprocessing/cub2002011/split_train_test.py \
 --data_dir '/users5/cpca082112025/tgoncalves/proto-counterfactuals/data/cub2002011-dataset'

echo 'Finished split train-val-test of CUB2002011.'