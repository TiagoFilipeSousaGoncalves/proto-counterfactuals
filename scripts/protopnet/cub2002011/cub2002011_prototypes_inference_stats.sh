#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=cub_infps                # Job name
#SBATCH -o cub_infps.out                  # STDOUT
#SBATCH -e cub_infps.err                  # STDERR



echo "CUB2002011 | Started | Prototype Inference Stats"

echo "CUB2002011 | ProtoPNet"
python src/protopnet/prototypes_inference_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2025-03-14_11-19-56/'

python src/protopnet/prototypes_inference_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2025-03-17_15-17-17/'

python src/protopnet/prototypes_inference_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2025-03-17_15-17-21/'

echo "CUB2002011 | Finished | Prototype Inference Stats"