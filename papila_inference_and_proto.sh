#!/bin/bash



echo "PAPILA | Started | Inference and Prototypes"

model="dppnet"

if [ $model == "ppnet" ]
then
    echo "PAPILA | ProtoPNet"
    python code/protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture densenet121 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/
    # python code/protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture densenet161 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/densenet161/XXX/
    python code/protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture resnet34 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/
    # python code/protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture resnet152 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/resnet152/XXX/
    python code/protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture vgg16 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/
    # python code/protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture vgg19 --num_workers 2 --gpu_id 0 --checkpoint papila/protopnet/vgg19/XXX/
elif [ $model == 'dppnet' ]
then
    echo "PAPILA | Deformable ProtoPNet"
    python code/deformable-protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/
    # python code/deformable-protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/densenet161/XXX/
    python code/deformable-protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/
    # python code/deformable-protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/resnet152/XXX/
    python code/deformable-protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/
    # python code/deformable-protopnet/models_inference_and_prototypes.py --dataset PAPILA --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "PAPILA | Finished | Inference and Prototypes"
