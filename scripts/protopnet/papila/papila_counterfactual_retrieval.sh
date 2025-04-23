#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                  # QOS
#SBATCH --job-name=pla_cntret              # Job name
#SBATCH -o pla_cntret.out                  # STDOUT
#SBATCH -e pla_cntret.err                  # STDERR



echo "PAPILA | Started | Counterfactual Retrieval"



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


echo "CUB2002011 | Finished | Counterfactual Retrieval"
