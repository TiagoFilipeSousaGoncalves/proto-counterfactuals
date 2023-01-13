#!/bin/bash



echo "PAPILA | Started | Counterfactual Retrieval"


model="ppnet"


if [ $model == "ppnet" ]
then
    echo "PAPILA | ProtoPNet | Convolution Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet161/XXX/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet152/XXX/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg19/XXX/ --generate_img_features --feature_space conv_features
    echo "PAPILA | ProtoPNet | Convolution Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet161/XXX/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet152/XXX/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg19/XXX/ --generate_img_features --feature_space proto_features
elif [ $model == 'dppnet' ]
then
    echo "PAPILA | Deformable ProtoPNet"
    # python code/protopnet_deform/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2022-12-06_18-16-44/ --generate_img_features
    # python code/protopnet_deform/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/2022-12-07_11-54-55/ --generate_img_features
    # python code/protopnet_deform/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2022-12-07_04-58-44/ --generate_img_features
    # python code/protopnet_deform/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/2022-12-07_15-56-33/ --generate_img_features
    # python code/protopnet_deform/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2022-12-07_17-48-57/ --generate_img_features
    # python code/protopnet_deform/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/2022-12-07_19-02-13/ --generate_img_features
else
    echo "Error"
fi

echo "PAPILA | Finished | Counterfactual Retrieval"
