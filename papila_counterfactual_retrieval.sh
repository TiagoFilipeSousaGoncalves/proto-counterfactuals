#!/bin/bash



echo "PAPILA | Started | Counterfactual Retrieval"


model="baseline"


if [ $model == "baseline" ]
then
    echo "PAPILA | Baseline | Convolution Feature Space"
    python code/baseline/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint papila/baseline/densenet121/2023-01-16_00-08-15/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint papila/baseline/densenet161/XXX/ --feature_space conv_features
    python code/baseline/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint papila/baseline/resnet34/2023-01-16_07-33-46/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint papila/baseline/resnet152/XXX/ --feature_space conv_features
    python code/baseline/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint papila/baseline/vgg16/2023-01-16_13-44-00/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint papila/baseline/vgg19/XXX/ --feature_space conv_features
elif [ $model == "ppnet" ]
then
    echo "PAPILA | ProtoPNet | Convolution Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet161/XXX/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet152/XXX/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg19/XXX/ --generate_img_features --feature_space conv_features
    echo "PAPILA | ProtoPNet | Prototype Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet121/2022-12-23_11-33-39/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/densenet161/XXX/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet34/2022-12-23_18-42-05/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/resnet152/XXX/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg16/2022-12-23_18-10-15/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint papila/protopnet/vgg19/XXX/ --generate_img_features --feature_space proto_features
elif [ $model == "dppnet" ]
then
    echo "PAPILA | Deformable ProtoPNet | Convolution Feature Space"
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/densenet161/XXX/ --generate_img_features --feature_space conv_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/resnet152/XXX/ --generate_img_features --feature_space conv_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/vgg19/XXX/ --generate_img_features --feature_space conv_features
    echo "PAPILA | Deformable ProtoPNet | Prototype Feature Space"
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/densenet161/XXX/ --generate_img_features --feature_space proto_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/resnet152/XXX/ --generate_img_features --feature_space proto_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PAPILA --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint papila/deformable-protopnet/vgg19/XXX/ --generate_img_features --feature_space proto_features
else
    echo "Error"
fi

echo "PAPILA | Finished | Counterfactual Retrieval"
