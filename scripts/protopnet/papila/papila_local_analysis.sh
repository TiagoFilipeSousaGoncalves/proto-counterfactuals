#!/bin/bash
#SBATCH -p debug_8gb                     # Partition
#SBATCH --qos=debug_8gb                       # QOS
#SBATCH --job-name=pla_lana                 # Job name
#SBATCH -o pla_lana.out                  # STDOUT
#SBATCH -e pla_lana.err                  # STDERR



echo "PAPILA | Started | Local Analysis"
echo "PAPILA | ProtoPNet"
python src/protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/'

python src/protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/'

python src/protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24'

echo "PAPILA | Finished | Local Analysis"