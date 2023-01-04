#!/bin/bash



echo "PAPILA | Started | Testing"


model="ppnet"


if [ $model == "ppnet" ]
then
    echo "PAPILA | ProtoPNet"
    python code/protopnet/models_test.py --dataset PAPILA --base_architecture densenet121 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/
    # python code/protopnet/models_test.py --dataset PAPILA --base_architecture densenet161 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/densenet161/XXX/
    python code/protopnet/models_test.py --dataset PAPILA --base_architecture resnet34 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/
    # python code/protopnet/models_test.py --dataset PAPILA --base_architecture resnet152 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/resnet152/XXX/
    python code/protopnet/models_test.py --dataset PAPILA --base_architecture vgg16 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/
    # python code/protopnet/models_test.py --dataset PAPILA --base_architecture vgg19 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/vgg19/XXX/
elif [ $model == 'dppnet' ]
then
    echo "PAPILA | Deformable ProtoPNet"
    # python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture densenet121 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2022-12-06_18-16-44/
    # python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture densenet161 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/2022-12-07_11-54-55/
    # python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture resnet34 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2022-12-07_04-58-44/
    # python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture resnet152 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/2022-12-07_15-56-33/
    # python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2022-12-07_17-48-57/
    # python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture vgg19 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/2022-12-07_19-02-13/
else
    echo "Error"
fi


echo "PAPILA | Finished | Testing"
