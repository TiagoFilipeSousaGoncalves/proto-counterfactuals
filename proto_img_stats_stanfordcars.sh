#!/bin/bash
#



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "STARTED | STANFORDCARS | Prototype Image Stats"

python code/prototypes_images_stats.py --checkpoint stanfordcars/densenet121/2022-10-24_08-55-48/
# python code/prototypes_images_stats.py --checkpoint 
python code/prototypes_images_stats.py --checkpoint stanfordcars/resnet34/2022-10-25_14-20-40/
# python code/prototypes_images_stats.py --checkpoint stanfordcars/resnet152
python code/prototypes_images_stats.py --checkpoint stanfordcars/vgg16/2022-10-26_10-49-44/
# python code/prototypes_images_stats.py --checkpoint stanfordcars/vgg19/2022-10-27_14-16-56/

echo "FINISHED | STANFORDCARS | Prototype Image Stats"
