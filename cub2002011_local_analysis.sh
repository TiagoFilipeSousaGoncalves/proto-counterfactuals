#!/bin/bash



echo "CUB2002011 | Started | Local Analysis"

model="ppnet"

if [ $model == "ppnet" ]
then
    echo "CUB2002011 | ProtoPNet"
    python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/densenet121/2022-09-14_06-32-51/
    # python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/densenet161/2022-09-15_16-14-45
    python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/resnet34/2022-09-17_17-03-33/
    # python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/resnet152/
    python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/vgg16/2022-10-14_21-14-35/
    # python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/vgg19/2022-10-17_08-13-40
elif [ $model == "dppnet" ]
then
    echo "CUB2002011 | Deformable ProtoPNet"
    python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/densenet121/2022-11-14_11-19-59/
    # python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/densenet161/
    python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/resnet34/2022-11-13_22-48-06/
    # python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/resnet152/
    python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/vgg16/2022-11-14_08-54-02/
    # python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/vgg19/
else
    echo "Error"
fi

echo "CUB2002011 | Finished | Local Analysis"
