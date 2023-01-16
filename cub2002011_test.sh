#!/bin/bash



echo "CUB2002011 | Started | Testing"

model="ppnet"

if [ $model = "ppnet" ]
then
    echo "CUB2002011 | ProtoPNet"
    python code/protopnet/models_test.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet121/2023-01-06_12-07-43/
    # python code/protopnet/models_test.py --dataset CUB2002011 --base_architecture densenet161 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet161/XXX/
    python code/protopnet/models_test.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet34/2022-12-29_19-34-24/
    # python code/protopnet/models_test.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet152/XXX/
    python code/protopnet/models_test.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg16/2022-12-30_22-45-59/
    # python code/protopnet/models_test.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg19/XXX/
elif [ $model == "dppnet" ]
then
    echo "CUB2002011 | Deformable ProtoPNet"
    python code/protopnet_deform/models_test.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/densenet121/2022-11-14_11-19-59/
    python code/protopnet_deform/models_test.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/resnet34/2022-11-13_22-48-06/
    python code/protopnet_deform/models_test.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/vgg16/2022-11-14_08-54-02/
else
    echo "Error"
fi

echo "FINISHED | CUB2002011 | TEST"
