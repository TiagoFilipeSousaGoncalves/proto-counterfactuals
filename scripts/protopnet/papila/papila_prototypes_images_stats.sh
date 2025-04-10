#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=pla_pis              # Job name
#SBATCH -o pla_pis.out                  # STDOUT
#SBATCH -e pla_pis.err                  # STDERR



echo "PAPILA | Started | Prototype Image Stats"
python src/protopnet/prototypes_images_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/'

python src/protopnet/prototypes_images_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/'

python src/protopnet/prototypes_images_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24'

echo "PAPILA | Finished | Prototype Image Stats"