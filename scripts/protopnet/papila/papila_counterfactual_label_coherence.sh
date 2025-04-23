#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=pla_plc              # Job name
#SBATCH -o pla_plc.out                  # STDOUT
#SBATCH -e pla_plc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"

echo "ProtoPNet"
echo "Counterfactual Label Coherence | PAPILA"
python src/protopnet/models_counterfactuals_labels_coherence.py \
 --dataset papila \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24/' \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa

python src/protopnet/models_counterfactuals_labels_coherence.py \
 --dataset papila \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/densenet121/2025-03-23_09-59-10/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/resnet34/2025-03-23_09-59-09/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/protopnet/vgg16/2025-03-23_15-12-24/' \
 --feature_space proto_features \
 --coherence_metric fleiss_kappa

echo "Counterfactual Label Coherence | Finished"