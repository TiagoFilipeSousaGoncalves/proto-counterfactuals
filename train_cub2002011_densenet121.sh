#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



echo "CUB2002011 DenseNet121"

python code/models_train.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 32 --num_workers 3 --gpu_id 0

echo "Finished"
