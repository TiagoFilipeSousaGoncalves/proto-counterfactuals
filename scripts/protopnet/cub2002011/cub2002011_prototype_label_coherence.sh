#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=pla_plc              # Job name
#SBATCH -o pla_plc.out                  # STDOUT
#SBATCH -e pla_plc.err                  # STDERR



echo "Prototype Label Coherence | Started"
echo "ProtoPNet"
echo "Prototype Label Coherence | CUB2002011"
python src/protopnet/prototypes_labels_coherence.py \
 --dataset cub2002011 \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2025-03-14_11-19-56/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2025-03-17_15-17-17/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2025-03-17_15-17-21/' \
 --coherence_metric earth_movers_distance
echo "Prototype Label Coherence | Finished"