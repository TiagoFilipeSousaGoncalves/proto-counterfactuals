#!/bin/bash
#SBATCH -p gpu_min8gb                   # Partition
#SBATCH --qos=gpu_min8gb                # QOS
#SBATCH --job-name=pla_clc              # Job name
#SBATCH -o pla_clc.out                  # STDOUT
#SBATCH -e pla_clc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"
echo "Baseline"
echo "Counterfactual Label Coherence | PAPILA"
python src/baseline/counterfactuals_labels_coherence.py \
 --dataset papila \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/densenet121/2025-03-19_22-34-07/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/resnet34/2025-03-19_23-07-26/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/papila/baseline/vgg16/2025-03-21_16-13-55/ \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa
echo "Counterfactual Label Coherence | Finished"