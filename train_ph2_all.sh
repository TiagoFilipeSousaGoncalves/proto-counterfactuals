#!/bin/bash



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"

echo "Submitting jobs"

# sbatch train_ph2_densenet121.sh 
sbatch train_ph2_densenet161.sh
sbatch train_ph2_resnet34.sh 
sbatch train_ph2_resnet152.sh
sbatch train_ph2_vgg16.sh
sbatch train_ph2_vgg19.sh

echo "All jobs submitted"
