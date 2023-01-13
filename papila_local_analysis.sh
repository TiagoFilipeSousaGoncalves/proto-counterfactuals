#!/bin/bash



echo "PAPILA | Started | Local Analysis"

model="ppnet"

if [ $model == "ppnet" ]
then
    echo "PAPILA | ProtoPNet"
    python code/protopnet/models_local_analysis.py --dataset PAPILA --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/
    # python code/protopnet/models_local_analysis.py --dataset PAPILA --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/
    python code/protopnet/models_local_analysis.py --dataset PAPILA --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/
    # python code/protopnet/models_local_analysis.py --dataset PAPILA --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/
    python code/protopnet/models_local_analysis.py --dataset PAPILA --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/
    # python code/protopnet/models_local_analysis.py --dataset PAPILA --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/
elif [ $model == 'dppnet' ]
then
    echo "PAPILA | Deformable ProtoPNet"
    # python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2022-12-18_22-05-03/
    # python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/2022-12-18_22-05-03/
    # python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2022-12-18_23-42-05/
    # python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/2022-12-18_22-09-25/
    # python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2022-12-19_00-02-17/
    # python code/protopnet_deform/models_local_analysis.py --dataset PH2 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/2022-12-19_00-47-32/
else
    echo "Error"
fi

echo "PAPILA | Finished | Local Analysis"
