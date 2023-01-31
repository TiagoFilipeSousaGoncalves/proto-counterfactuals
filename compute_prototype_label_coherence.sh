#!/bin/bash

echo "Prototype Label Coherence | Started"

model="ppnet"

if [ $model=="ppnet" ]
then
    echo "ProtoPNet"
    echo "Prototype Label Coherence | CUB2002011"
    python code/protopnet/prototypes_labels_coherence.py --dataset CUB2002011 --append-checkpoints cub2002011/protopnet/densenet121/2023-01-06_12-07-43/ --append-checkpoints cub2002011/protopnet/resnet34/2022-12-29_19-34-24/ --append-checkpoints cub2002011/protopnet/vgg16/2022-12-30_22-45-59/

    echo "Prototype Label Coherence | PAPILA"
    python code/protopnet/prototypes_labels_coherence.py --dataset PAPILA --append-checkpoints papila/protopnet/densenet121/2022-12-23_11-33-39/ --append-checkpoints papila/protopnet/resnet34/2022-12-23_18-42-05/ --append-checkpoints papila/protopnet/vgg16/2022-12-23_18-10-15/

    echo "Prototype Label Coherence | PH2"
    python code/protopnet/prototypes_labels_coherence.py --dataset PH2 --append-checkpoints ph2/protopnet/densenet121/2022-12-06_15-51-53/ --append-checkpoints ph2/protopnet/resnet34/2022-12-06_22-45-55/ --append-checkpoints ph2/protopnet/vgg16/2022-12-07_00-42-57/
elif [$model=="dppnet"]
then
    echo "Deformable ProtoPNet"
else
    echo "Error"
fi

echo "Prototype Label Coherence | Finished"
