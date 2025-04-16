#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=pla_plc              # Job name
#SBATCH -o pla_plc.out                  # STDOUT
#SBATCH -e pla_plc.err                  # STDERR



echo "Prototype Label Coherence | Started"
echo "Prototype Label Coherence | PAPILA"
python src/protopnet/prototypes_labels_coherence.py \
 --dataset papila \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24' \
 --coherence_metric earth_movers_distance
echo "Prototype Label Coherence | Finished"