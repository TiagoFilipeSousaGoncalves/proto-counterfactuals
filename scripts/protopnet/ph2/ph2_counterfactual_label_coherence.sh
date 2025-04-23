#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=ph2_plc              # Job name
#SBATCH -o ph2_plc.out                  # STDOUT
#SBATCH -e ph2_plc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"

echo "ProtoPNet"
echo "Counterfactual Label Coherence | PH2"
python src/protopnet/models_counterfactuals_labels_coherence.py \
 --dataset ph2 \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/densenet121/2025-03-24_16-19-14/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/resnet34/2025-03-24_18-30-11/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/vgg16/2025-03-24_19-06-46/' \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa

python src/protopnet/models_counterfactuals_labels_coherence.py \
 --dataset ph2 \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/densenet121/2025-03-24_16-19-14/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/resnet34/2025-03-24_18-30-11/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/vgg16/2025-03-24_19-06-46/' \
 --feature_space proto_features \
 --coherence_metric fleiss_kappa

echo "Counterfactual Label Coherence | Finished"