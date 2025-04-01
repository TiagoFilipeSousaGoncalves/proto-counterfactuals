#!/bin/bash



echo "CUB2002011 | Started | Inference and Prototypes"

model="dppnet"

if [ $model == "ppnet" ]
then
    echo "CUB2002011 | ProtoPNet"
    python code/protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture densenet121 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet121/2023-01-06_12-07-43/
    # python code/protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture densenet161 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet161/XXX/
    python code/protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture resnet34 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet34/2022-12-29_19-34-24/
    # python code/protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture resnet152 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet152/XXX/
    python code/protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture vgg16 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg16/2022-12-30_22-45-59/
    # python code/protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture vgg19 --num_workers 2 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg19/XXX/
elif [ $model == "dppnet" ]
then
    echo "CUB2002011 | Deformable ProtoPNet"
    python code/deformable-protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/
    # python code/deformable-protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/densenet161/XXX/
    python code/deformable-protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/resnet34/2023-01-11_18-27-35/
    # python code/deformable-protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/resnet152/XXX/
    python code/deformable-protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/vgg16/2023-01-13_07-25-59/
    # python code/deformable-protopnet/models_inference_and_prototypes.py --dataset CUB2002011 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "CUB2002011 | Finished | Inference and Prototypes"
