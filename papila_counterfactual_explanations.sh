#!/bin/bash



echo "Started | PAPILA | Counterfactual Explanations"


model="ppnet"


if [ $model == "ppnet" ]
then
    echo "PAPILA | ProtoPNet"
    python code/protopnet/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/
    # python code/protopnet/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/densenet161/XXX/
    python code/protopnet/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/
    # python code/protopnet/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/resnet152/XXX/
    python code/protopnet/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/
    # python code/protopnet/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/protopnet/vgg19/XXX/
elif [ $model == 'dppnet' ]
then
    echo "PAPILA | Deformable ProtoPNet"
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/densenet121/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/densenet161/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/resnet34/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/resnet152/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/vgg16/XXX/
    # python code/protopnet_deform/counterfactuals_explanations.py --dataset PAPILA --checkpoint papila/deformable-protopnet/vgg19/XXX/
else
    echo "Error"
fi

echo "Finished | PH2 | Counterfactual Explanations"
