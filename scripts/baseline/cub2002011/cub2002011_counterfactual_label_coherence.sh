#!/bin/bash

echo "Counterfactual Label Coherence | Started"

model="baseline"

if [ $model == "baseline" ]
then
    echo "Baseline"
    echo "Counterfactual Label Coherence | CUB2002011"
    python code/baseline/counterfactuals_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/baseline/densenet121/2023-03-28_14-00-03/ --append-checkpoints cub2002011/baseline/resnet34/2023-03-29_23-07-14/ --append-checkpoints cub2002011/baseline/vgg16/2023-03-29_23-07-14/ --feature_space conv_features --coherence_metric fleiss_kappa

    echo "Counterfactual Label Coherence | PAPILA"
    python code/baseline/counterfactuals_labels_coherence.py --dataset PAPILA --append-checkpoints papila/baseline/densenet121/2023-03-28_13-59-11/ --append-checkpoints papila/baseline/resnet34/2023-03-28_13-59-15/ --append-checkpoints papila/baseline/vgg16/2023-03-28_13-59-24 --feature_space conv_features --coherence_metric fleiss_kappa

    echo "Counterfactual Label Coherence | PH2"
    python code/baseline/counterfactuals_labels_coherence.py --dataset PH2 --append-checkpoints ph2/baseline/densenet121/2023-03-28_13-59-31/ --append-checkpoints ph2/baseline/resnet34/2023-03-28_18-16-50/ --append-checkpoints ph2/baseline/vgg16/2023-03-29_15-36-45/ --feature_space conv_features --coherence_metric fleiss_kappa
elif [ $model == "ppnet" ]
then
    echo "ProtoPNet"
    echo "Counterfactual Label Coherence | CUB2002011"
    python code/protopnet/counterfactuals_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --append-checkpoints cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --append-checkpoints cub2002011/protopnet/vgg16/2022-12-30_22-45-59/ --feature_space conv_features --coherence_metric fleiss_kappa
    python code/protopnet/counterfactuals_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --append-checkpoints cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --append-checkpoints cub2002011/protopnet/vgg16/2022-12-30_22-45-59/ --feature_space proto_features --coherence_metric fleiss_kappa

    echo "Counterfactual Label Coherence | PAPILA"
    python code/protopnet/counterfactuals_labels_coherence.py --dataset PAPILA --append-checkpoints papila/protopnet/densenet121/2022-12-23_11-33-39/ --append-checkpoints papila/protopnet/resnet34/2022-12-23_18-42-05/ --append-checkpoints papila/protopnet/vgg16/2022-12-23_18-10-15/ --feature_space conv_features --coherence_metric fleiss_kappa
    python code/protopnet/counterfactuals_labels_coherence.py --dataset PAPILA --append-checkpoints papila/protopnet/densenet121/2022-12-23_11-33-39/ --append-checkpoints papila/protopnet/resnet34/2022-12-23_18-42-05/ --append-checkpoints papila/protopnet/vgg16/2022-12-23_18-10-15/ --feature_space proto_features --coherence_metric fleiss_kappa

    echo "Counterfactual Label Coherence | PH2"
    python code/protopnet/counterfactuals_labels_coherence.py --dataset PH2 --append-checkpoints ph2/protopnet/densenet121/2022-12-06_15-51-53/ --append-checkpoints ph2/protopnet/resnet34/2022-12-06_22-45-55/ --append-checkpoints ph2/protopnet/vgg16/2022-12-07_00-42-57/ --feature_space conv_features --coherence_metric fleiss_kappa
    python code/protopnet/counterfactuals_labels_coherence.py --dataset PH2 --append-checkpoints ph2/protopnet/densenet121/2022-12-06_15-51-53/ --append-checkpoints ph2/protopnet/resnet34/2022-12-06_22-45-55/ --append-checkpoints ph2/protopnet/vgg16/2022-12-07_00-42-57/ --feature_space proto_features --coherence_metric fleiss_kappa
elif [ $model == "dppnet" ]
then
    echo "Deformable ProtoPNet"
    echo "Counterfactual Label Coherence | CUB2002011"
    python code/deformable-protopnet/counterfactuals_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/ --append-checkpoints cub2002011/deformable-protopnet/resnet34/2023-01-11_18-27-35/ --append-checkpoints cub2002011/deformable-protopnet/vgg16/2023-01-13_07-25-59/ --feature_space conv_features --coherence_metric fleiss_kappa
    python code/deformable-protopnet/counterfactuals_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/ --append-checkpoints cub2002011/deformable-protopnet/resnet34/2023-01-11_18-27-35/ --append-checkpoints cub2002011/deformable-protopnet/vgg16/2023-01-13_07-25-59/ --feature_space proto_features --coherence_metric fleiss_kappa

    echo "Counterfactual Label Coherence | PAPILA"
    python code/deformable-protopnet/counterfactuals_labels_coherence.py --dataset PAPILA --append-checkpoints papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/ --append-checkpoints papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/ --append-checkpoints papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/ --feature_space conv_features --coherence_metric fleiss_kappa
    python code/deformable-protopnet/counterfactuals_labels_coherence.py --dataset PAPILA --append-checkpoints papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/ --append-checkpoints papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/ --append-checkpoints papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/ --feature_space proto_features --coherence_metric fleiss_kappa

    echo "Counterfactual Label Coherence | PH2"
    python code/deformable-protopnet/counterfactuals_labels_coherence.py --dataset PH2 --append-checkpoints ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/ --append-checkpoints ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/ --append-checkpoints ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/ --feature_space conv_features --coherence_metric fleiss_kappa
    python code/deformable-protopnet/counterfactuals_labels_coherence.py --dataset PH2 --append-checkpoints ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/ --append-checkpoints ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/ --append-checkpoints ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/ --feature_space proto_features --coherence_metric fleiss_kappa
else
    echo "Error"
fi

echo "Counterfactual Label Coherence | Finished"
