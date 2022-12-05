#!/bin/bash



# CUB2002011 "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
echo "Started Local Analysis | CUB2002011"


# ProtoPNet
# echo "ProtoPNet"
# python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/densenet121/2022-09-14_06-32-51/
# python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet161 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/densenet161/2022-09-15_16-14-45
# python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/resnet34/2022-09-17_17-03-33/
# python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/resnet152/
# python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/vgg16/2022-10-14_21-14-35/
# python code/protopnet/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/vgg19/2022-10-17_08-13-40



# Deformable ProtoPNet
echo "Deformable ProtoPNet"
python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet121 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/densenet121/2022-11-14_11-19-59/
python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture densenet161 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/densenet161/
python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet34 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/resnet34/2022-11-13_22-48-06/
python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture resnet152 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/resnet152/
python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg16 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/vgg16/2022-11-14_08-54-02/
python code/protopnet_deform/models_local_analysis.py --dataset CUB2002011 --base_architecture vgg19 --batchsize 16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint results/cub2002011/deformable-protopnet/vgg19/

echo "Finished Local Analysis | CUB2002011"
