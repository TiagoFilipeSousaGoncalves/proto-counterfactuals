#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=pla_infp                 # Job name
#SBATCH -o pla_infp.out                  # STDOUT
#SBATCH -e pla_infp.err                  # STDERR



echo "PAPILA | Started | Inference and Prototypes"

echo "PAPILA | ProtoPNet"
python src/protopnet/models_inference_and_prototypes.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/'

python src/protopnet/models_inference_and_prototypes.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/'

python src/protopnet/models_inference_and_prototypes.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24'

echo "PAPILA | Finished | Inference and Prototypes"