#!/bin/bash
#SBATCH -p gpu_min8gb                   # Partition
#SBATCH --qos=gpu_min8gb                # QOS
#SBATCH --job-name=cub_clc              # Job name
#SBATCH -o cub_clc.out                  # STDOUT
#SBATCH -e cub_clc.err                  # STDERR



echo "Counterfactual Label Coherence | Started"
echo "Baseline"
echo "Counterfactual Label Coherence | CUB2002011"
python src/baseline/models_counterfactuals_labels_coherence.py \
 --dataset cub2002011 \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/densenet121/2025-03-12_01-24-26/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/resnet34/2025-03-12_09-25-24/ \
 --result_dir_list /nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/cub2002011/baseline/vgg16/2025-03-14_00-14-55/ \
 --feature_space conv_features \
 --coherence_metric fleiss_kappa
echo "Counterfactual Label Coherence | Finished"