#!/bin/bash
#SBATCH -p cpu_8cores                     # Partition
#SBATCH --qos=cpu_8cores                       # QOS
#SBATCH --job-name=ph2_infps                # Job name
#SBATCH -o ph2_infps.out                  # STDOUT
#SBATCH -e ph2_infps.err                  # STDERR



echo "Started | PH2 | Prototype Inference Stats"
echo "PH2 | ProtoPNet"
python src/protopnet/prototypes_inference_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/densenet121/2025-03-24_16-19-14/'

python src/protopnet/prototypes_inference_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/resnet34/2025-03-24_18-30-11/'

python src/protopnet/prototypes_inference_stats.py \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/vgg16/2025-03-24_19-06-46/'
echo "Finished | PH2 | Prototype Inference Stats"