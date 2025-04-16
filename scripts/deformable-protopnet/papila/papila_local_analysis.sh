#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=pla_lana                 # Job name
#SBATCH -o pla_lana.out                  # STDOUT
#SBATCH -e pla_lana.err                  # STDERR



echo "PAPILA | Started | Local Analysis"
echo "PAPILA | Deformable ProtoPNet"
python src/deformable-protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture densenet121 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/densenet121/2025-03-26_11-31-33/

python src/deformable-protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture resnet34 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/resnet34/2025-03-26_12-12-55/

python src/deformable-protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/papila-dataset-glaucoma-fundus-images' \
 --dataset papila \
 --base_architecture vgg16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/vgg16/2025-03-26_11-31-32/
echo "PAPILA | Finished | Local Analysis"