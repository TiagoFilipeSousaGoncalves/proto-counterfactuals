#!/bin/bash



# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"

echo "Submitting jobs | CUB2002011"

sbatch train_cub2002011_densenet121.sh
sbatch train_cub2002011_densenet161.sh
sbatch train_cub2002011_resnet34.sh
sbatch train_cub2002011_resnet152.sh
sbatch train_cub2002011_vgg16.sh
sbatch train_cub2002011_vgg19.sh

echo "All jobs submitted | CUB2002011"
