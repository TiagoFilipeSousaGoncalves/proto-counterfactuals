#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=cub_pis              # Job name
#SBATCH -o cub_pis.out                  # STDOUT
#SBATCH -e cub_pis.err                  # STDERR



echo "CUB2002011 | Started | Prototype Image Stats"
echo "CUB2002011 | Deformable ProtoPNet"
python src/deformable-protopnet/prototypes_images_stats.py \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2025-03-25_20-10-27/

python src/deformable-protopnet/prototypes_images_stats.py \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2025-03-25_20-54-57/

python src/deformable-protopnet/prototypes_images_stats.py \
 --results_dir /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2025-03-25_18-54-27/
echo "CUB2002011 | Finished | Prototype Image Stats"