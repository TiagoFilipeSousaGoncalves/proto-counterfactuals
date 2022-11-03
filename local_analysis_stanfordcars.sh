#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=la_stanfordcars        # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STANFORDCARS"

# python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture densenet121 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/densenet121/2022-10-24_08-55-48/ --compute_metrics
# python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture densenet161 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/densenet161 --compute_metrics
# python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture resnet34 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/resnet34/2022-10-25_14-20-40/ --compute_metrics
# python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture resnet152 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/resnet152 --compute_metrics
# python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture vgg16 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/vgg16/2022-10-26_10-49-44/ --compute_metrics
python code/models_local_analysis.py --dataset STANFORDCARS --base_architecture vgg19 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint stanfordcars/vgg19/2022-10-27_14-16-56/ --compute_metrics

echo "Finished."
