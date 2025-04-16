#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=ph2_lana                 # Job name
#SBATCH -o ph2_lana.out                  # STDOUT
#SBATCH -e ph2_lana.err                  # STDERR



echo "PH2 | Started | Local Analysis"
echo "PH2 | Deformable ProtoPNet"
python src/deformable-protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2025-03-25_20-10-27/

python src/deformable-protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2025-03-25_20-54-57/

python src/deformable-protopnet/models_local_analysis.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --subtractive_margin \
 --using_deform \
 --last_layer_fixed \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2025-03-25_18-54-27/
echo "PH2 | Finished | Local Analysis"