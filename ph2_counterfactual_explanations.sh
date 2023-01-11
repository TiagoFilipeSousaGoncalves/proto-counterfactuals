#!/bin/bash



echo "Started | PH2 | Counterfactual Explanations"


model="ppnet"


if [ $model == "ppnet" ]
then
    echo "PH2 | ProtoPNet | Convolution Feature Space"
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/ --feature_space conv_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/ --feature_space conv_features
    echo "PH2 | ProtoPNet | Prototype Feature Space"
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/ --feature_space proto_features
    python code/protopnet/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/ --feature_space proto_features
elif [ $model == 'dppnet' ]
then
    echo "PH2 | Deformable ProtoPNet"
    python code/protopnet_deform/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/ --feature_space conv_features
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/deformable-protopnet/densenet161/XXX/ --feature_space conv_features
    python code/protopnet_deform/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/ --feature_space conv_features
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/deformable-protopnet/resnet152/XXX/ --feature_space conv_features
    python code/protopnet_deform/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/ --feature_space conv_features
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PH2 --checkpoint ph2/deformable-protopnet/vgg19/XXX/ --feature_space conv_features
else
    echo "Error"
fi

echo "Finished | PH2 | Counterfactual Explanations"
