#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=la_stanfordcars        # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STANFORDCARS"

python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture densenet121 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/densenet121/2022-09-20_16-50-50/ --compute_metrics
# python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture densenet161 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/densenet161 --compute_metrics
python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture resnet34 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/resnet34/2022-09-20_16-50-50/ --compute_metrics
# python code/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/resnet152 --compute_metrics
# python code/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/vgg16 --compute_metrics
# python code/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 3 --gpu_id 0 --checkpoint stanfordcars/vgg19 --compute_metrics

echo "Finished."
