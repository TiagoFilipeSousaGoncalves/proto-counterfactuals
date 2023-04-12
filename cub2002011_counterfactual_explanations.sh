#!/bin/bash



echo "Started | CUB2002011 | Counterfactual Explanations"


model="dppnet"


if [ $model == "ppnet" ]
then
    echo "CUB2002011 | ProtoPNet | Convolution Feature Space"
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --feature_space conv_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/densenet161/XXX/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --feature_space conv_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/resnet152/XXX/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/vgg16/2022-12-30_22-45-59/ --feature_space conv_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/vgg19/XXX/ --feature_space conv_features
    echo "CUB2002011 | ProtoPNet | Prototype Feature Space"
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --feature_space proto_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/densenet161/XXX/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --feature_space proto_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/resnet152/XXX/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/vgg16/2022-12-30_22-45-59/ --feature_space proto_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/protopnet/vgg19/XXX/ --feature_space proto_features
elif [ $model == 'dppnet' ]
then
    echo "CUB2002011 | Deformable ProtoPNet | Convolution Feature Space"
    python code/deformable-protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/ --feature_space conv_features
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/densenet161/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/resnet34/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/resnet152/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/vgg16/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/vgg19/XXX/
    
    # echo "CUB2002011 | Deformable ProtoPNet | Prototype Feature Space"
    # python code/deformable-protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/ --feature_space proto_features
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/densenet161/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/resnet34/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/resnet152/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/vgg16/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "Finished | PH2 | Counterfactual Explanations"
