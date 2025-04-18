# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/local_analysis.py
# Note: Finding the nearest prototypes to a test image

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# PyTorch Imports
import torch
from torch.autograd import Variable
import torch.utils.data


# Project Imports
from data_utilities import undo_preprocess, save_prototype, save_prototype_self_activation, save_prototype_original_img_with_bbox, imsave_with_bbox, save_preprocessed_img
from prototypes_utilities import find_high_activation_crop




# Function: Retrieve prototypes from an image
def retrieve_image_prototypes(
        image_analysis_path, 
        prototypes_img_dir, 
        ppnet_model, 
        device, 
        eval_transforms, 
        eval_image_path, 
        eval_image_label, 
        norm_params, 
        img_size,
        most_k_activated=10,
        save_original_images=False,
        save_prototypes=False):

    # Open a file to save a small report w/ .TXT extension
    if os.path.exists(os.path.join(image_analysis_path, "report.txt")):
        os.remove(os.path.join(image_analysis_path, "report.txt"))
    report = open(os.path.join(image_analysis_path, "report.txt"), "at")

    # Get the directory of the saved prototypes
    # saved_prototypes_dir = os.path.join(weights_dir, 'prototypes')
    # saved_prototypes_dir -> prototypes_img_dir

    # Get prototype information and confirm prototype class identity
    prototype_info = np.load(os.path.join(prototypes_img_dir, 'bb.npy'))
    prototype_img_identity = prototype_info[:, -1]


    # Append information to the report
    report.write('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.\n')
    report.write('Their class identities are: ' + str(prototype_img_identity) + '\n')


    # Confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet_model.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()

    # Number of prototypes that connect to their classe identities:
    nr_prototypes_cls_ident = np.sum(prototype_max_connection == prototype_img_identity)

    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet_model.num_prototypes:
        report.write('All prototypes connect most strongly to their respective classes.\n')
    else:
        report.write('WARNING: Not all prototypes connect most strongly to their respective classes.\n')


    # Get protoype shape
    prototype_shape = ppnet_model.prototype_shape

    # Get max distance
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]



    # Load the image and labels
    eval_image_name = eval_image_path.split('/')[-1]
    img_pil = Image.open(eval_image_path).convert('RGB')
    img_tensor = eval_transforms(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_eval = img_variable.to(device)
    # labels_test = torch.tensor([test_image_label])

    # Run inference with ppnet
    logits, min_distances = ppnet_model(images_eval)
    _, distances = ppnet_model.push_forward(images_eval)
    prototype_activations = ppnet_model.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet_model.distance_2_similarity(distances)
    if ppnet_model.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist


    # Create tables of predictions and ground-truth
    # tables = []
    # for i in range(logits.size(0)):
        # tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
        # print(str(i) + ' ' + str(tables[-1]))
        # report.write(f"{str(i)}, {str(tables[-1])}\n")
    
    # Get prediction and ground-truth labels
    # idx = 0
    # predicted_cls = tables[0][0]
    # correct_cls = tables[0][1]
    

    # Apply Softmax to obtain scaled logits
    s_logits = torch.nn.Softmax(dim=1)(logits)

    # Sort indices to obtain the prediction
    sorted_indices = torch.argsort(s_logits, dim=1)
    

    # Get prediction
    predicted_cls = int(sorted_indices[0][-1].item())
    correct_cls = int(eval_image_label)


    # Append information to the report
    report.write(f'Predicted: {str(predicted_cls)}\n')
    report.write(f'Actual: {str(correct_cls)}\n')
    
    
    # Save original image
    if save_original_images:
        save_preprocessed_img(
            fname=os.path.join(image_analysis_path, 'original_img.png'),
            preprocessed_imgs=images_eval,
            mean=norm_params["mean"],
            std=norm_params["std"],
            index=0
        )

    # Get original image
    original_img = undo_preprocess(images_eval, norm_params["mean"], norm_params["std"])
    original_img = original_img[0]
    original_img = original_img.detach().cpu().numpy()
    original_img = np.transpose(original_img, (1,2,0))


    # Get most activated (nearest) K prototypes of this image
    if not os.path.isdir(os.path.join(image_analysis_path, 'most_activated_prototypes')):
        os.makedirs(os.path.join(image_analysis_path, 'most_activated_prototypes'))

    # Append information to the report
    report.write(f'Most activated {most_k_activated} prototypes of this image:\n')
    array_act, sorted_indices_act = torch.sort(prototype_activations[0])


    # Create a list of the identitities of the top-K most activated prototypes
    topk_proto_cls_ident = list()


    # for i in range(1, 11):
    for i in range(1, most_k_activated + 1):
        report.write(f'Top-{i} activated prototype for this image:\n')


        # Save prototype
        prototype_fname = os.path.join(image_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype.png' % i)
        report.write(f"Prototype: {prototype_fname}\n")
        if save_prototypes:
            save_prototype(
                prototype_fname=prototype_fname,
                prototypes_img_dir=prototypes_img_dir,
                index=sorted_indices_act[-i].item()
            )


        # Save prototype original image with bounding-box
        prototype_bbox_fname = os.path.join(image_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype_in_original_pimg.png' % i)
        report.write(f"Prototype with bounding-box: {prototype_bbox_fname}\n")
        if save_prototypes:
            save_prototype_original_img_with_bbox(
                prototype_bbox_fname=prototype_bbox_fname,
                prototypes_img_dir=prototypes_img_dir,
                index=sorted_indices_act[-i].item(),
                bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                color=(0, 255, 255)
            )


        # Save prototype self-activation
        prototype_self_act_fname = os.path.join(image_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype_self_act.png' % i)
        report.write(f"Prototype self-activation: {prototype_self_act_fname}\n")
        if save_prototypes:
            save_prototype_self_activation(
                prototype_self_act_fname=prototype_self_act_fname,
                prototypes_img_dir=prototypes_img_dir,
                index=sorted_indices_act[-i].item()
            )


        # Add information to the report
        report.write(f'Prototype index: {sorted_indices_act[-i].item()}\n')
        report.write(f'Prototype class identity: {prototype_img_identity[sorted_indices_act[-i].item()]}\n')
        
        # Append it to the list
        topk_proto_cls_ident.append(prototype_img_identity[sorted_indices_act[-i].item()])


        # Prototype maximum connection
        if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
            report.write(f'Prototype connection identity: {prototype_max_connection[sorted_indices_act[-i].item()]}\n')


        # Add more information to the report
        report.write(f'Activation value (similarity score): {array_act[-i]}\n')
        report.write(f'Last layer connection with predicted class: {ppnet_model.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]}\n')


        # Get activation pattern and upsampled activation pattern
        activation_pattern = prototype_activation_patterns[0][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)

        # Show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
        high_act_patch_fname = os.path.join(image_analysis_path, 'most_activated_prototypes', 'most_highly_activated_patch_by_top-%d_prototype.png' % i)
        report.write(f'Most highly activated patch of the chosen image by this prototype: {high_act_patch_fname}\n')
        if save_prototypes:
            plt.imsave(high_act_patch_fname, high_act_patch)


        # Show the most highly activated patch of the image by this prototype in the original image
        high_act_patch_img_bbox_fname = os.path.join(image_analysis_path, 'most_activated_prototypes', 'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i)
        report.write(f'Most highly activated patch by this prototype shown in the original image: {high_act_patch_img_bbox_fname}\n')
        if save_prototypes:
            imsave_with_bbox(
                high_act_patch_img_bbox_fname=high_act_patch_img_bbox_fname,
                img_rgb=original_img,
                bbox_height_start=high_act_patch_indices[0],
                bbox_height_end=high_act_patch_indices[1],
                bbox_width_start=high_act_patch_indices[2],
                bbox_width_end=high_act_patch_indices[3],
                color=(0, 255, 255)
            )


        # Show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        prototype_act_map_fname = os.path.join(image_analysis_path, 'most_activated_prototypes', 'prototype_activation_map_by_top-%d_prototype.png' % i)
        report.write(f'Prototype activation map of the chosen image: {prototype_act_map_fname}\n')
        if save_prototypes:
            plt.imsave(prototype_act_map_fname, overlayed_img)
        report.write('\n')


    # Close report
    report.close()


    return eval_image_name, correct_cls, predicted_cls, nr_prototypes_cls_ident, topk_proto_cls_ident



# FIXME: Get prototypes from top-K classes
def get_prototypes_from_topk_classes(k, logits, idx, ppnet_model, image_analysis_path, prototypes_img_dir, prototype_activations, prototype_info, prototype_img_identity, prototype_max_connection, prototype_activation_patterns, img_size, original_img, predicted_cls, correct_cls):

    # Get prototypes from top-K classes
    k = 50
    print('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[idx], k=k)
    for i, c in enumerate(topk_classes.detach().cpu().numpy()):
        # TODO: makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))
        if not os.path.isdir(os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1))):
            os.makedirs(os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1)))

        print('top %d predicted class: %d' % (i+1, c))
        print('logit of the class: %f' % topk_logits[i])
        class_prototype_indices = np.nonzero(ppnet_model.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_cnt = 1
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]

            # Save prototype
            save_prototype(
                fname=os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_activated_prototype.png' % prototype_cnt),
                prototypes_img_dir=prototypes_img_dir,
                epoch=None,
                index=prototype_index
            )
        

            # Save prototype original image with bounding-box
            save_prototype_original_img_with_bbox(
                fname=os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                prototypes_img_dir=prototypes_img_dir,
                epoch=None,
                index=prototype_index,
                bbox_height_start=prototype_info[prototype_index][1],
                bbox_height_end=prototype_info[prototype_index][2],
                bbox_width_start=prototype_info[prototype_index][3],
                bbox_width_end=prototype_info[prototype_index][4],
                color=(0, 255, 255)
            )


            # Save prototype self-activation
            save_prototype_self_activation(
                fname=os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                prototypes_img_dir=prototypes_img_dir,
                epoch=None,
                index=prototype_index
            )


            print('prototype index: {0}'.format(prototype_index))
            print('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
            
            if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                print('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
            
            print('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
            print('last layer connection: {0}'.format(ppnet_model.last_layer.weight[c][prototype_index]))
            
            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
            
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
            print('most highly activated patch of the chosen image by this prototype:')
            # plt.axis('off')
            plt.imsave(os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1), 'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt), high_act_patch)
            print('most highly activated patch by this prototype shown in the original image:')
            
            # Save image with bounding-box
            imsave_with_bbox(
                fname=os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1), 'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                img_rgb=original_img,
                bbox_height_start=high_act_patch_indices[0],
                bbox_height_end=high_act_patch_indices[1],
                bbox_width_start=high_act_patch_indices[2],
                bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255)
            )
            
            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            print('prototype activation map of the chosen image:')
            # plt.axis('off')
            plt.imsave(os.path.join(image_analysis_path, 'top-%d_class_prototypes' % (i+1), 'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt), overlayed_img)
            print('--------------------------------------------------------------')
            prototype_cnt += 1
        print('***************************************************************')


    # Check if predicted class is the correct class
    if predicted_cls == correct_cls:
        print('Prediction is correct.')
    else:
        print('Prediction is wrong.')


    return
