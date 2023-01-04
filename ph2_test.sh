#!/bin/bash



echo "PH2 | Started | Testing"


model="ppnet"


if [ $model == "ppnet" ]
then
    echo "PH2 | ProtoPNet"
    python code/protopnet/models_test.py --dataset PH2 --base_architecture densenet121 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/
    python code/protopnet/models_test.py --dataset PH2 --base_architecture densenet161 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/
    python code/protopnet/models_test.py --dataset PH2 --base_architecture resnet34 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/
    python code/protopnet/models_test.py --dataset PH2 --base_architecture resnet152 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/
    python code/protopnet/models_test.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/
    python code/protopnet/models_test.py --dataset PH2 --base_architecture vgg19 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/
elif [ $model == 'dppnet' ]
then
    echo "PH2 | Deformable ProtoPNet"
    python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture densenet121 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2022-12-06_18-16-44/
    python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture densenet161 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/2022-12-07_11-54-55/
    python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture resnet34 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2022-12-07_04-58-44/
    python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture resnet152 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/2022-12-07_15-56-33/
    python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture vgg16 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2022-12-07_17-48-57/
    python code/protopnet_deform/models_test.py --dataset PH2 --base_architecture vgg19 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/2022-12-07_19-02-13/
else
    echo "Error"
fi


echo "PH2 | Finished | Testing"
