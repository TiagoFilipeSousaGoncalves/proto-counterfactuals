#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=pla_plc              # Job name
#SBATCH -o pla_plc.out                  # STDOUT
#SBATCH -e pla_plc.err                  # STDERR



echo "Prototype Label Coherence | Started"
echo "Prototype Label Coherence | CUB2002011"
python src/deformable-protopnet/prototypes_labels_coherence.py \
 --dataset cub2002011 \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/densenet121/2025-03-24_09-39-04/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/resnet34/2025-03-24_23-56-33/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/vgg16/2025-03-24_23-59-23/' \
 --coherence_metric earth_movers_distance
echo "Prototype Label Coherence | Finished"