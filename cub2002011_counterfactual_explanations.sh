#!/bin/bash



echo "Started | CUB2002011 | Counterfactual Explanations"


model="ppnet"


if [ $model == "ppnet" ]
then
    echo "CUB2002011 | ProtoPNet | Convolution Feature Space"
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --feature_space conv_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/densenet161/XXX/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --feature_space conv_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/resnet152/XXX/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --feature_space conv_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/vgg19/XXX/ --feature_space conv_features
    echo "CUB2002011 | ProtoPNet | Prototype Feature Space"
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --feature_space proto_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/densenet161/XXX/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --feature_space proto_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/resnet152/XXX/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --feature_space proto_features
    # python code/protopnet/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/protopnet/vgg19/XXX/ --feature_space proto_features
elif [ $model == 'dppnet' ]
then
    echo "CUB2002011 | Deformable ProtoPNet"
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/densenet121/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/densenet161/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/resnet34/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/resnet152/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/vgg16/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset CUB2002011 --checkpoint papila/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "Finished | PH2 | Counterfactual Explanations"
