#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=pla_plc              # Job name
#SBATCH -o pla_plc.out                  # STDOUT
#SBATCH -e pla_plc.err                  # STDERR



echo "Prototype Label Coherence | Started"
echo "Prototype Label Coherence | PAPILA"
python src/deformable-protopnet/prototypes_labels_coherence.py \
 --dataset papila \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/densenet121/2025-03-26_11-31-33/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/resnet34/2025-03-26_12-12-55/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/vgg16/2025-03-26_11-31-32/' \
 --coherence_metric earth_movers_distance
echo "Prototype Label Coherence | Finished"