#!/bin/bash



echo "CUB2002011 | Started | Prototype Image Stats"

model="ppnet"

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
    # python code/protopnet_deform/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/densenet121/2022-12-06_18-16-44/
    # python code/protopnet_deform/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/densenet161/2022-12-07_11-54-55/
    # python code/protopnet_deform/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/resnet34/2022-12-07_04-58-44/
    # python code/protopnet_deform/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/resnet152/2022-12-07_15-56-33/
    # python code/protopnet_deform/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/vgg16/2022-12-07_17-48-57/
    # python code/protopnet_deform/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/vgg19/2022-12-07_19-02-13/
else
    echo "Error"
fi

echo "CUB2002011 | Finished | Prototype Image Stats"
