#!/bin/bash



echo "PAPILA | Started | Prototype Image Stats"

model="dppnet"

if [ $model == "ppnet" ]
then
echo "PAPILA | ProtoPNet"
    python code/protopnet/prototypes_images_stats.py --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/
    # python code/protopnet/prototypes_images_stats.py --checkpoint papila/protopnet/densenet161/XXX/
    python code/protopnet/prototypes_images_stats.py --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/
    # python code/protopnet/prototypes_images_stats.py --checkpoint papila/protopnet/resnet152/XXX/
    python code/protopnet/prototypes_images_stats.py --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/
    # python code/protopnet/prototypes_images_stats.py --checkpoint papila/protopnet/vgg19/XXX/
elif [ $model == 'dppnet' ]
then
    echo "PAPILA | Deformable ProtoPNet"
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint papila/deformable-protopnet/densenet161/XXX/
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint papila/deformable-protopnet/resnet152/XXX/
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint papila/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "PAPILA | Finished | Prototype Image Stats"
