#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB                 # Partition        (check w/ $sinfo)
#SBATCH --job-name=ga_cub2002011          # Job name
#SBATCH -o slurm.%N.%j.out                # STDOUT
#SBATCH -e slurm.%N.%j.err                # STDERR



# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STARTED | CUB2002011 | Prototype Inference Stats"

# python code/prototypes_inference_stats.py --checkpoint cub2002011/densenet121/2022-09-14_06-32-51/
# python code/prototypes_inference_stats.py --checkpoint cub2002011/densenet161/2022-09-15_16-14-45/
# python code/prototypes_inference_stats.py --checkpoint cub2002011/resnet34/2022-09-17_17-03-33/
<<<<<<< HEAD
# python code/prototypes_images_stats.py --dataset CUB2002011 --base_architecture resnet152 --checkpoint cub2002011/resnet152/2022-08-19_18-07-45/
# python code/prototypes_images_stats.py --checkpoint cub2002011/vgg16/2022-10-14_21-14-35/
python code/prototypes_images_stats.py --checkpoint cub2002011/vgg19/2022-10-17_08-13-40
=======
# python code/prototypes_inference_stats.py --checkpoint cub2002011/resnet152/
# python code/prototypes_inference_stats.py --checkpoint cub2002011/vgg16/2022-10-14_21-14-35/
python code/prototypes_inference_stats.py --checkpoint cub2002011/vgg19/2022-10-17_08-13-40/
>>>>>>> 46c4f8815fd2e7d5bf79584ab924f0975fcc6a73

echo "FINISHED | CUB2002011 | Prototype Inference Stats"
