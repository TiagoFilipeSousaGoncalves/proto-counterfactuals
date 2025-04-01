#!/bin/bash
#SBATCH -p debug_8gb                     # Partition
#SBATCH --qos=debug_8gb                       # QOS
#SBATCH --job-name=cub_lana                 # Job name
#SBATCH -o cub_lana.out                  # STDOUT
#SBATCH -e cub_lana.err                  # STDERR



echo "CUB2002011 | Started | Local Analysis"
echo "CUB2002011 | ProtoPNet"
python code/protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
 --dataset cub2002011 \
 --base_architecture densenet121 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2025-03-14_11-19-56/'

# python code/protopnet/models_local_analysis.py \
#  --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
#  --dataset cub2002011 \
#  --base_architecture resnet34 \
#  --num_workers 4 \
#  --gpu_id 0 \
#  --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2025-03-17_15-17-17/'

# python code/protopnet/models_local_analysis.py \
#  --data_dir '/nas-ctm01/datasets/public/cub2002011-dataset' \
#  --dataset cub2002011 \
#  --base_architecture vgg16 \
#  --num_workers 4 \
#  --gpu_id 0 \
#  --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2025-03-17_15-17-21/'

echo "CUB2002011 | Finished | Local Analysis"
