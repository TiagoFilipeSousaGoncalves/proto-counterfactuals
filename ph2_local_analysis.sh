#!/bin/bash



# PH2 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "Started | PH2 | Local Analysis"

if [ $1 == "ppnet" ]
then
    echo "PH2 | ProtoPNet"
    python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/
    python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/
    python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/
    python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/
    python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/
    python code/protopnet/models_local_analysis.py --dataset PH2 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/
elif [ $1 == 'dppnet' ]
then
    echo "PH2 | Deformable ProtoPNet"
    python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2022-12-06_18-16-44/
    python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/2022-12-07_11-54-55/
    python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2022-12-07_04-58-44/
    python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/2022-12-07_15-56-33/
    python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2022-12-07_17-48-57/
    python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/2022-12-07_19-02-13/
else
    echo "Error"
fi

echo "Finished | PH2 | Local Analysis"
