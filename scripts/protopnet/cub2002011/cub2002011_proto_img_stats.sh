#!/bin/bash



echo "CUB2002011 | Started | Prototype Image Stats"

model="dppnet"

if [ $model == "ppnet" ]
then
echo "CUB2002011 | ProtoPNet"
    python code/protopnet/prototypes_images_stats.py --checkpoint cub2002011/protopnet/densenet121/2023-01-06_12-07-43/
    # python code/protopnet/prototypes_images_stats.py --checkpoint cub2002011/protopnet/densenet161/XXX/
    python code/protopnet/prototypes_images_stats.py --checkpoint cub2002011/protopnet/resnet34/2022-12-29_19-34-24/
    # python code/protopnet/prototypes_images_stats.py --checkpoint cub2002011/protopnet/resnet152/XXX/
    python code/protopnet/prototypes_images_stats.py --checkpoint cub2002011/protopnet/vgg16/2022-12-30_22-45-59/
    # python code/protopnet/prototypes_images_stats.py --checkpoint cub2002011/protopnet/vgg19/XXX/
elif [ $model == 'dppnet' ]
then
    echo "CUB2002011 | Deformable ProtoPNet"
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint cub2002011/deformable-protopnet/densenet161/XXX/
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint cub2002011/deformable-protopnet/resnet34/2023-01-11_18-27-35/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint cub2002011/deformable-protopnet/resnet152/XXX/
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint cub2002011/deformable-protopnet/vgg16/2023-01-13_07-25-59/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint cub2002011/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "CUB2002011 | Finished | Prototype Image Stats"
