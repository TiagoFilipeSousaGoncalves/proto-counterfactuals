#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=cub_plc              # Job name
#SBATCH -o cub_plc.out                  # STDOUT
#SBATCH -e cub_plc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"

echo "ProtoPNet"
echo "Counterfactual Label Coherence | CUB2002011"
python src/protopnet/models_counterfactuals_labels_coherence.py \
 --dataset cub2002011 \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2025-03-14_11-19-56/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2025-03-17_15-17-17/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2025-03-17_15-17-21/' \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa

python src/protopnet/models_counterfactuals_labels_coherence.py \
 --dataset cub2002011 \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/densenet121/2025-03-14_11-19-56/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/resnet34/2025-03-17_15-17-17/' \
 --result_dir_list '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/protopnet/vgg16/2025-03-17_15-17-21/' \
 --feature_space proto_features \
 --coherence_metric fleiss_kappa

echo "Counterfactual Label Coherence | Finished"