#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                  # QOS
#SBATCH --job-name=cub_infps              # Job name
#SBATCH -o cub_infps.out                  # STDOUT
#SBATCH -e cub_infps.err                  # STDERR



echo "CUB2002011 | Started | Prototype Inference Stats"
echo "CUB2002011 | Deformable ProtoPNet"
python src/deformable-protopnet/prototypes_inference_stats.py \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/densenet121/2025-03-26_11-31-33/

python src/deformable-protopnet/prototypes_inference_stats.py \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/resnet34/2025-03-26_12-12-55/

python src/deformable-protopnet/prototypes_inference_stats.py \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/vgg16/2025-03-26_11-31-32/
echo "CUB2002011 | Finished | Prototype Inference Stats"