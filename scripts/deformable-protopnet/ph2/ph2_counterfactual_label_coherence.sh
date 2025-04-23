#!/bin/bash
#SBATCH -p debug_8gb                   # Partition
#SBATCH --qos=debug_8gb                # QOS
#SBATCH --job-name=ph2_clc              # Job name
#SBATCH -o ph2_clc.out                  # STDOUT
#SBATCH -e ph2_clc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"

echo "Deformable ProtoPNet"
echo "Counterfactual Label Coherence | PH2"
python src/deformable-protopnet/models_counterfactuals_labels_coherence.py \
 --dataset ph2 \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2025-03-25_20-10-27/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2025-03-25_20-54-57/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2025-03-25_18-54-27/ \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa

python src/deformable-protopnet/models_counterfactuals_labels_coherence.py \
 --dataset ph2 \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/densenet121/2025-03-25_20-10-27/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/resnet34/2025-03-25_20-54-57/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/deformable-protopnet/vgg16/2025-03-25_18-54-27/ \
 --feature_space proto_features \
 --coherence_metric fleiss_kappa

echo "Counterfactual Label Coherence | Finished"