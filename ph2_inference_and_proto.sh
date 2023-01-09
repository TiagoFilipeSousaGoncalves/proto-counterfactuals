#!/bin/bash



echo "Started | PH2 | Inference and Prototypes"


model="dppnet"


if [ $model == "ppnet" ]
then
    echo "PH2 | ProtoPNet"
    python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture densenet121 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/
    python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture densenet161 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/
    python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture resnet34 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/
    python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture resnet152 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/
    python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture vgg16 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/
    python code/protopnet/models_inference_and_prototypes.py --dataset PH2 --base_architecture vgg19 --num_workers 2 --gpu_id 0 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/
elif [ $model == 'dppnet' ]
then
    echo "PH2 | Deformable ProtoPNet"
    python code/protopnet_deform/models_inference_and_prototypes.py --dataset PH2 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/
    # python code/protopnet_deform/models_inference_and_prototypes.py --dataset PH2 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/XXX/
    python code/protopnet_deform/models_inference_and_prototypes.py --dataset PH2 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/
    # python code/protopnet_deform/models_inference_and_prototypes.py --dataset PH2 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/XXX/
    python code/protopnet_deform/models_inference_and_prototypes.py --dataset PH2 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/
    # python code/protopnet_deform/models_inference_and_prototypes.py --dataset PH2 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "Finished | PH2 | Inference and Prototypes"
