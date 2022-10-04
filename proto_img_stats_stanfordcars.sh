#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=ga_cub2002011          # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STARTED ! STANFORDCARS | Prototype Image Stats"

python code/prototypes_images_stats.py --checkpoint stanfordcars/densenet121/2022-09-20_16-50-50/
# python code/prototypes_images_stats.py --checkpoint 
python code/prototypes_images_stats.py --checkpoint stanfordcars/resnet34/2022-09-20_16-50-50/
# python code/prototypes_images_stats.py --dataset CUB2002011 --base_architecture resnet152 --checkpoint 
# python code/prototypes_images_stats.py --dataset CUB2002011 --base_architecture vgg16 --checkpoint 
# python code/prototypes_images_stats.py --dataset CUB2002011 --base_architecture vgg19 --checkpoint 

echo "FINISHED ! STANFORDCARS | Prototype Image Stats"
