#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=ph2_plc              # Job name
#SBATCH -o ph2_plc.out                  # STDOUT
#SBATCH -e ph2_plc.err                  # STDERR



echo "Prototype Label Coherence | Started"
echo "Prototype Label Coherence | PH2"
python src/deformable-protopnet/prototypes_labels_coherence.py \
 --dataset ph2 \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2025-03-25_20-10-27/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2025-03-25_20-54-57/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2025-03-25_18-54-27/' \
 --coherence_metric earth_movers_distance
echo "Prototype Label Coherence | Finished"