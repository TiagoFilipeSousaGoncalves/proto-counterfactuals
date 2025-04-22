#!/bin/bash
#SBATCH -p cpu_8cores                   # Partition
#SBATCH --qos=cpu_8cores                # QOS
#SBATCH --job-name=ph2_clc              # Job name
#SBATCH -o ph2_clc.out                  # STDOUT
#SBATCH -e ph2_clc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"
echo "Baseline"
echo "Counterfactual Label Coherence | PH2"
python src/baseline/counterfactuals_labels_coherence.py \
 --dataset ph2 \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/densenet121/2025-03-24_11-14-24/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/resnet34/2025-03-24_11-59-55/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/baseline/vgg16/2025-03-24_14-36-33/ \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa
echo "Counterfactual Label Coherence | Finished"