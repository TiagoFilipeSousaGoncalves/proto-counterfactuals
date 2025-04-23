#!/bin/bash
#SBATCH -p debug_8gb                   # Partition
#SBATCH --qos=debug_8gb                # QOS
#SBATCH --job-name=pla_clc              # Job name
#SBATCH -o pla_clc.out                  # STDOUT
#SBATCH -e pla_clc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"

echo "Deformable ProtoPNet"
echo "Counterfactual Label Coherence | PAPILA"
python src/deformable-protopnet/models_counterfactuals_labels_coherence.py \
 --dataset papila \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/densenet121/2025-03-26_11-31-33/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/resnet34/2025-03-26_12-12-55/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/vgg16/2025-03-26_11-31-32/ \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa

python src/deformable-protopnet/models_counterfactuals_labels_coherence.py \
 --dataset papila \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/densenet121/2025-03-26_11-31-33/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/resnet34/2025-03-26_12-12-55/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/deformable-protopnet/vgg16/2025-03-26_11-31-32/ \
 --feature_space proto_features \
 --coherence_metric fleiss_kappa

echo "Counterfactual Label Coherence | Finished"