# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/prune.py

# Imports
import os
import shutil
from collections import Counter
import numpy as np

# PyTorch Imports
import torch

# Project Imports
from analysis_utilities import find_k_nearest_patches_to_prototypes



# Function: Prune prototypes
def prune_prototypes(dataloader, prototype_network_parallel, device, k, prune_threshold, preprocess_input_function, original_model_dir, epoch_number, copy_prototype_imgs=True):

    # Run global analysis
    nearest_train_patch_class_ids = find_k_nearest_patches_to_prototypes(
        dataloader=dataloader,
        prototype_network_parallel=prototype_network_parallel,
        device=device,
        k=k,
        preprocess_input_function=preprocess_input_function,
        full_save=False
    )


    # Find prototypes to prune
    # TODO: original_num_prototypes = prototype_network_parallel.module.num_prototypes
    original_num_prototypes = prototype_network_parallel.num_prototypes

    # Create list of prototypes to prune
    prototypes_to_prune = []
    
    # Go through the number of prototypes of network
    # for j in range(prototype_network_parallel.module.num_prototypes):
    for j in range(prototype_network_parallel.num_prototypes):
        # class_j = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
        class_j = torch.argmax(prototype_network_parallel.prototype_class_identity[j]).item()
        nearest_train_patch_class_counts_j = Counter(nearest_train_patch_class_ids[j])
        
        # if no such element is in Counter, it will return 0
        if nearest_train_patch_class_counts_j[class_j] < prune_threshold:
            prototypes_to_prune.append(j)


    # Log prints
    print('k = {}, prune_threshold = {}'.format(k, prune_threshold))
    print('{} prototypes will be pruned'.format(len(prototypes_to_prune)))


    # Bookkeeping of prototypes to be pruned
    # class_of_prototypes_to_prune = torch.argmax(prototype_network_parallel.module.prototype_class_identity[prototypes_to_prune], dim=1).numpy().reshape(-1, 1)
    class_of_prototypes_to_prune = torch.argmax(prototype_network_parallel.prototype_class_identity[prototypes_to_prune], dim=1).numpy().reshape(-1, 1)
    prototypes_to_prune_np = np.array(prototypes_to_prune).reshape(-1, 1)
    prune_info = np.hstack((prototypes_to_prune_np, class_of_prototypes_to_prune))


    # TODO: Validate uppon review
    if epoch_number:
        if not os.path.isdir(os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number, k, prune_threshold))):
            os.makedirs(os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number, k, prune_threshold)))

        np.save(os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number, k, prune_threshold), 'prune_info.npy'), prune_info)
    
    else:
        if not os.path.isdir(os.path.join(original_model_dir, 'pruned_prototypes_k{}_pt{}'.format(k, prune_threshold))):
            os.makedirs(os.path.join(original_model_dir, 'pruned_prototypes_k{}_pt{}'.format(k, prune_threshold)))

        np.save(os.path.join(original_model_dir, 'pruned_prototypes_k{}_pt{}'.format(k, prune_threshold), 'prune_info.npy'), prune_info)



    # Prune prototypes
    # prototype_network_parallel.module.prune_prototypes(prototypes_to_prune)
    prototype_network_parallel.prune_prototypes(prototypes_to_prune)
    # torch.save(obj=prototype_network_parallel.module, f=os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number, k, prune_threshold), model_name + '-pruned.pth'))

    if copy_prototype_imgs:
        if epoch_number:
            original_img_dir = os.path.join(original_model_dir, 'img', 'epoch-%d' % epoch_number)
            dst_img_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number, k, prune_threshold), 'img', 'epoch-%d' % epoch_number)
        
        else:
            original_img_dir = os.path.join(original_model_dir, 'prototypes')
            dst_img_dir = os.path.join(original_model_dir, 'pruned_prototypes_k{}_pt{}'.format(k, prune_threshold), 'prototypes')


        # Create destination directory
        if not os.path.isdir(dst_img_dir):
            os.makedirs(dst_img_dir)


        # Get prototypes to keep
        prototypes_to_keep = list(set(range(original_num_prototypes)) - set(prototypes_to_prune))


        # Iterate through prototypes to keep
        for idx in range(len(prototypes_to_keep)):
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img%d.png' % prototypes_to_keep[idx]), dst=os.path.join(dst_img_dir, 'prototype-img%d.png' % idx))
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img-original%d.png' % prototypes_to_keep[idx]), dst=os.path.join(dst_img_dir, 'prototype-img-original%d.png' % idx))
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img-original_with_self_act%d.png' % prototypes_to_keep[idx]), dst=os.path.join(dst_img_dir, 'prototype-img-original_with_self_act%d.png' % idx))
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-self-act%d.npy' % prototypes_to_keep[idx]), dst=os.path.join(dst_img_dir, 'prototype-self-act%d.npy' % idx))

            if epoch_number:
                bb = np.load(os.path.join(original_img_dir, 'bb%d.npy' % epoch_number))
                bb = bb[prototypes_to_keep]
                np.save(os.path.join(dst_img_dir, 'bb%d.npy' % epoch_number), bb)

                bb_rf = np.load(os.path.join(original_img_dir, 'bb-receptive_field%d.npy' % epoch_number))
                bb_rf = bb_rf[prototypes_to_keep]
                np.save(os.path.join(dst_img_dir, 'bb-receptive_field%d.npy' % epoch_number), bb_rf)

            else:
                bb = np.load(os.path.join(original_img_dir, 'bb.npy'))
                bb = bb[prototypes_to_keep]
                np.save(os.path.join(dst_img_dir, 'bb.npy'), bb)

                bb_rf = np.load(os.path.join(original_img_dir, 'bb-receptive_field.npy'))
                bb_rf = bb_rf[prototypes_to_keep]
                np.save(os.path.join(dst_img_dir, 'bb-receptive_field.npy'), bb_rf)


    return prune_info
