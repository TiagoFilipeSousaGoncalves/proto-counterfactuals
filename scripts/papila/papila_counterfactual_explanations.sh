#!/bin/bash



echo "Started | PAPILA | Counterfactual Explanations"


model="dppnet"


if [ $model == "ppnet" ]
then
    echo "PAPILA | ProtoPNet | Convolution Feature Space"
    python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --feature_space conv_features
    # python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/densenet161/XXX/ --feature_space conv_features
    python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --feature_space conv_features
    # python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/resnet152/XXX/ --feature_space conv_features
    python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --feature_space conv_features
    # python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/vgg19/XXX/ --feature_space conv_features
    echo "PAPILA | ProtoPNet | Prototype Feature Space"
    python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --feature_space proto_features
    # python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/densenet161/XXX/ --feature_space proto_features
    python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --feature_space proto_features
    # python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/resnet152/XXX/ --feature_space proto_features
    python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --feature_space proto_features
    # python code/protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/vgg19/XXX/ --feature_space proto_features
elif [ $model == "dppnet" ]
then
    echo "PAPILA | Deformable ProtoPNet | Convolutional Feature Space"
    python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/ --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/densenet161/XXX/ --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/ --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/resnet152/XXX/ --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/ --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/vgg19/XXX/ --feature_space conv_features
    
    # echo "PAPILA | Deformable ProtoPNet | Prototype Feature Space"
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/ --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/densenet161/XXX/ --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/ --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/resnet152/XXX/ --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/ --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/vgg19/XXX/ --feature_space proto_features
else
    echo "Error"
fi

echo "Finished | PAPILA | Counterfactual Explanations"
