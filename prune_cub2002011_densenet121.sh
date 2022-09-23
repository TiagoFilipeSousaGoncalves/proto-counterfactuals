#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=tr_stan_dn161          # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR
#SBATCH --reservation=tgoncalv_1          # Reservation Name



# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"

echo "Prunning | CUB2002011:DenseNet121 | STARTED"

python code/models_prototype_pruning.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 32 --optimize_last_layer --num_workers o --gpu_id 0 --checkpoint cub2002011/densenet121/2022-09-14_06-32-51/ --k 6 --prune_threshold 3

echo "Prunning | CUB2002011 | FINISHED"
