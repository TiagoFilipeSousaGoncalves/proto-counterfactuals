#!/bin/bash



echo "CUB2002011 | Started | Counterfactual Retrieval"


model="baseline"


if [ $model == "baseline" ]
then
    echo "CUB2002011 | Baseline | Convolution Feature Space"
    python code/baseline/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet121 --num_workers 4 --gpu_id 0 --checkpoint cub2002011/baseline/densenet121/2023-03-28_14-00-03/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/baseline/densenet161/XXX/ --feature_space conv_features
    python code/baseline/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet34 --num_workers 4 --gpu_id 0 --checkpoint cub2002011/baseline/resnet34/2023-03-29_23-07-14/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/baseline/resnet152/XXX/ --feature_space conv_features
    python code/baseline/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg16 --num_workers 4 --gpu_id 0 --checkpoint cub2002011/baseline/vgg16/2023-03-29_23-07-14/ --feature_space conv_features
    # python code/baseline/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/baseline/vgg19/XXX/ --feature_space conv_features
elif [ $model == "ppnet" ]
then
    echo "CUB2002011 | ProtoPNet | Convolution Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet161/XXX/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet152/XXX/ --generate_img_features --feature_space conv_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg16/2022-12-30_22-45-59/ --generate_img_features --feature_space conv_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg19/XXX/ --generate_img_features --feature_space conv_features
    echo "CUB2002011 | ProtoPNet | Prototype Feature Space"
    python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet121 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet161 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/densenet161/XXX/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet34 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet152 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/resnet152/XXX/ --generate_img_features --feature_space proto_features
    python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg16 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg16/2022-12-30_22-45-59/ --generate_img_features --feature_space proto_features
    # python code/protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg19 --num_workers 0 --gpu_id 0 --checkpoint cub2002011/protopnet/vgg19/XXX/ --generate_img_features --feature_space proto_features
elif [ $model == 'dppnet' ]
then
    echo "CUB2002011 | Deformable ProtoPNet | Convolution Feature Space"
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/densenet161/XXX/ --generate_img_features --feature_space conv_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/resnet34/2023-01-11_18-27-35/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/resnet152/XXX/ --generate_img_features --feature_space conv_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/vgg16/2023-01-13_07-25-59/ --generate_img_features --feature_space conv_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/vgg19/XXX/ --generate_img_features --feature_space conv_features
    echo "CUB2002011 | Deformable ProtoPNet | Prototype Feature Space"
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet121 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture densenet161 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/densenet161/XXX/ --generate_img_features --feature_space proto_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet34 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/resnet34/2023-01-11_18-27-35/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture resnet152 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/resnet152/XXX/ --generate_img_features --feature_space proto_features
    python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg16 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/vgg16/2023-01-13_07-25-59/ --generate_img_features --feature_space proto_features
    # python code/deformable-protopnet/models_counterfactuals_retrieval.py --dataset CUB2002011 --base_architecture vgg19 --subtractive_margin --using_deform --last_layer_fixed --num_workers 0 --gpu_id 0 --checkpoint cub2002011/deformable-protopnet/vgg19/XXX/ --generate_img_features --feature_space proto_features
else
    echo "Error"
fi

echo "CUB2002011 | Finished | Counterfactual Retrieval"
