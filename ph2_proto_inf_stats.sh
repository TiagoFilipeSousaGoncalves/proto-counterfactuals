#!/bin/bash



echo "Started | PH2 | Prototype Inference Stats"


model="dppnet"


if [ $model == "ppnet" ]
then
    echo "PH2 | ProtoPNet"
    python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/
    python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/
    python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/
    python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/
    python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/
    python code/protopnet/prototypes_inference_stats.py --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/
elif [ $model == 'dppnet' ]
then
    echo "PH2 | Deformable ProtoPNet"
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/densenet161/XXX/
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/resnet152/XXX/
    python code/deformable-protopnet/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/
    # python code/deformable-protopnet/prototypes_images_stats.py --checkpoint ph2/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "Finished | PH2 | Prototype Inference Stats"
