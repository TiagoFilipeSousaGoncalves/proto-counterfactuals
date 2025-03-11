#!/bin/bash



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"

echo "Submitting jobs"

sbatch train_stanfordcars_densenet121.sh
# sbatch train_stanfordcars_densenet161.sh
sbatch train_stanfordcars_resnet34.sh
# sbatch train_stanfordcars_resnet152.sh
# sbatch train_stanfordcars_vgg16.sh
# sbatch train_stanfordcars_vgg19.sh

echo "All jobs submitted"
