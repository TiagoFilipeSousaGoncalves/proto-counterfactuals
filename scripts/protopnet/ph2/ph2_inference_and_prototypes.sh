#!/bin/bash
#SBATCH -p gpu_min11gb                     # Partition
#SBATCH --qos=gpu_min11gb                       # QOS
#SBATCH --job-name=ph2_infp                 # Job name
#SBATCH -o ph2_infp.out                  # STDOUT
#SBATCH -e ph2_infp.err                  # STDERR



echo "Started | PH2 | Inference and Prototypes"
echo "PH2 | ProtoPNet"
python src/protopnet/models_inference_and_prototypes.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture densenet121 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/densenet121/2025-03-24_16-19-14/'

python src/protopnet/models_inference_and_prototypes.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture resnet34 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/resnet34/2025-03-24_18-30-11/'

python src/protopnet/models_inference_and_prototypes.py \
 --data_dir '/nas-ctm01/datasets/public/MEDICAL/ph2-database' \
 --dataset ph2 \
 --base_architecture vgg16 \
 --num_workers 4 \
 --gpu_id 0 \
 --results_dir '/nas-ctm01/homes/tgoncalv/proto-counterfactuals/results/ph2/protopnet/vgg16/2025-03-24_19-06-46/'
echo "Finished | PH2 | Inference and Prototypes"