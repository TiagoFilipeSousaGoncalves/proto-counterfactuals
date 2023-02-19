#!/bin/bash



echo "PH2 | Started | Counterfactual Retrieval"


model="baseline"

if [ $model == "baseline" ]
then
     --append-checkpoints  --append-checkpoints 
    echo "PH2 | Baseline | Convolution Feature Space"
    python code/baseline/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint ph2/baseline/densenet121/2023-01-16_20-22-43/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint ph2/baseline/densenet161/XXX/ --feature_space conv_features
    python code/baseline/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint ph2/baseline/resnet34/2023-01-16_22-56-35/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint ph2/baseline/resnet152/XXX/ --feature_space conv_features
    python code/baseline/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint ph2/baseline/vgg16/2023-01-17_01-29-53/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint ph2/baseline/vgg19/XXX/ --feature_space conv_features
elif [ $model == "ppnet" ]
then
    echo "PH2 | ProtoPNet | Convolution Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/ --generate_img_features --feature_space conv_features
    echo "PH2 | ProtoPNet | Prototype Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet121/2022-12-06_15-51-53/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/densenet161/2022-12-06_19-46-07/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/resnet34/2022-12-06_22-45-55/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/resnet152/2022-12-07_00-40-00/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/vgg16/2022-12-07_00-42-57/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint ph2/protopnet/vgg19/2022-12-07_02-48-40/ --generate_img_features --feature_space proto_features
elif [ $model == "dppnet" ]
then
    echo "PH2 | Deformable ProtoPNet | Convolution Feature Space"
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/XXX/ --generate_img_features --feature_space conv_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/XXX/ --generate_img_features --feature_space conv_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/XXX/ --generate_img_features --feature_space conv_features
    echo "PH2 | Deformable ProtoPNet | Prototype Feature Space"
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/densenet161/XXX/ --generate_img_features --feature_space proto_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/resnet152/XXX/ --generate_img_features --feature_space proto_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset PH2 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint ph2/deformable-protopnet/vgg19/XXX/ --generate_img_features --feature_space proto_features
else
    echo "Error"
fi

echo "PH2 | Finished | Counterfactual Retrieval"
