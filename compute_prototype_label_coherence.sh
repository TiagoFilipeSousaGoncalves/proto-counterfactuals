#!/bin/bash

echo "Prototype Label Coherence | Started"

model="dppnet"

if [ $model == "ppnet" ]
then
    echo "ProtoPNet"
    echo "Prototype Label Coherence | CUB2002011"
    python code/protopnet/prototypes_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --append-checkpoints cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --append-checkpoints cub2002011/protopnet/vgg16/2022-12-30_22-45-59/ --coherence_metric earth_movers_distance

    echo "Prototype Label Coherence | PAPILA"
    python code/protopnet/prototypes_labels_coherence.py --dataset PAPILA --append-checkpoints papila/protopnet/densenet121/2022-12-23_11-33-39/ --append-checkpoints papila/protopnet/resnet34/2022-12-23_18-42-05/ --append-checkpoints papila/protopnet/vgg16/2022-12-23_18-10-15/ --coherence_metric earth_movers_distance

    echo "Prototype Label Coherence | PH2"
    python code/protopnet/prototypes_labels_coherence.py --dataset PH2 --append-checkpoints ph2/protopnet/densenet121/2022-12-06_15-51-53/ --append-checkpoints ph2/protopnet/resnet34/2022-12-06_22-45-55/ --append-checkpoints ph2/protopnet/vgg16/2022-12-07_00-42-57/ --coherence_metric earth_movers_distance
elif [ $model == "dppnet" ]
then
    echo "Deformable ProtoPNet"
    echo "Prototype Label Coherence | CUB2002011"
    python code/deformable-protopnet/prototypes_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/deformable-protopnet/densenet121/2023-01-09_01-07-48/ --append-checkpoints cub2002011/deformable-protopnet/resnet34/2023-01-11_18-27-35/ --append-checkpoints cub2002011/deformable-protopnet/vgg16/2023-01-13_07-25-59/ --coherence_metric earth_movers_distance

    echo "Prototype Label Coherence | PAPILA"
    python code/deformable-protopnet/prototypes_labels_coherence.py --dataset PAPILA --append-checkpoints papila/deformable-protopnet/densenet121/2023-01-04_12-12-15/ --append-checkpoints papila/deformable-protopnet/resnet34/2023-01-04_16-02-21/ --append-checkpoints papila/deformable-protopnet/vgg16/2023-01-04_18-47-51/ --coherence_metric earth_movers_distance

    echo "Prototype Label Coherence | PH2"
    python code/deformable-protopnet/prototypes_labels_coherence.py --dataset PH2 --append-checkpoints ph2/deformable-protopnet/densenet121/2023-01-02_08-43-56/ --append-checkpoints ph2/deformable-protopnet/resnet34/2023-01-02_10-08-37/ --append-checkpoints ph2/deformable-protopnet/vgg16/2023-01-04_10-43-58/ --coherence_metric earth_movers_distance
else
    echo "Error"
fi

echo "Prototype Label Coherence | Finished"
