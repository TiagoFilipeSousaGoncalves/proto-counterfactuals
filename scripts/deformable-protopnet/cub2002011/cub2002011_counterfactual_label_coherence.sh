#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=cub_clc              # Job name
#SBATCH -o cub_clc.out                  # STDOUT
#SBATCH -e cub_clc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"

echo "Deformable ProtoPNet"
echo "Counterfactual Label Coherence | CUB2002011"
python src/deformable-protopnet/models_counterfactuals_labels_coherence.py \
 --dataset cub2002011 \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/densenet121/2025-03-24_09-39-04/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/resnet34/2025-03-24_23-56-33/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/vgg16/2025-03-24_23-59-23/ \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa

python src/deformable-protopnet/models_counterfactuals_labels_coherence.py \
 --dataset cub2002011 \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/densenet121/2025-03-24_09-39-04/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/resnet34/2025-03-24_23-56-33/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/deformable-protopnet/vgg16/2025-03-24_23-59-23/ \
 --feature_space proto_features \
 --coherence_metric fleiss_kappa

echo "Counterfactual Label Coherence | Finished"