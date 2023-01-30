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
from data_utilities import save_preprocessed_img, save_prototype, save_prototype_box, save_deform_info, imsave_with_bbox
from prototypes_utilities import find_high_activation_crop, get_deformation_info



# Function: Retrieve prototypes from an image
def retrieve_image_prototypes(save_analysis_path, weights_dir, ppnet_model, device, test_transforms, test_image_dir, test_image_name, test_image_label, img_size, most_k_activated=10, save_image=False):

    # Open a file to save a small report w/ .TXT extension
    report_path = os.path.join(save_analysis_path, "report.txt")
    if os.path.exists(report_path):
        os.remove(report_path)
    report = open(report_path, "at")

    # Specify the test image to be analyzed
    test_image_path = os.path.join(test_image_dir, test_image_name)


    # TODO: Review this path
    # load_img_dir = os.path.join(load_model_dir, 'img')
    saved_prototypes_dir = os.path.join(weights_dir, 'prototypes')

    # Get Prototype Information
    prototype_info = np.load(os.path.join(saved_prototypes_dir, 'bb.npy'))
    prototype_img_identity = prototype_info[:, -1]

    # Append this information to the report
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

    
    # Load image and labels
    img_pil = Image.open(test_image_path).convert("RGB")
    img_tensor = test_transforms(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.to(device)


    # Run inference with deform-ppnet
    logits, additional_returns = ppnet_model(images_test)
    prototype_activations = additional_returns[3]
    conv_output, prototype_activation_patterns = ppnet_model.push_forward(images_test)


    # Get offsets
    offsets, _ = get_deformation_info(conv_output, ppnet_model, device)
    offsets = offsets.detach()


    # # Create tables of predictions and ground-truth
    # tables = []
    # for i in range(logits.size(0)):
    #     tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    #     report.write(str(i) + ' ' + str(tables[-1]) + '\n')


    # # Get prediction and ground-truth labels
    # idx = 0
    # predicted_cls = tables[idx][0]
    # correct_cls = tables[idx][1]

    # Apply Softmax to obtain scaled logits
    s_logits = torch.nn.Softmax(dim=1)(logits)

    # Sort indices to obtain the prediction
    sorted_indices = torch.argsort(s_logits, dim=1)
    

    # Get prediction
    predicted_cls = int(sorted_indices[0][-1].item())
    correct_cls = int(test_image_label)
    
    # Append this information to the report
    report.write(f'Predicted: {str(predicted_cls)}\n')
    report.write(f'Actual: {str(correct_cls)}\n')

    # Get original image
    original_img = save_preprocessed_img(
        fname=os.path.join(save_analysis_path, 'original_img.png'),
        preprocessed_imgs=images_test,
        index=0,
        save_img=False
    )



    # Get most activated (nearest) K prototypes of this image
    if not os.path.isdir(os.path.join(save_analysis_path, 'most_activated_prototypes')):
        os.makedirs(os.path.join(save_analysis_path, 'most_activated_prototypes'))


    # Append information to the report
    report.write(f'Most activated {most_k_activated} prototypes of this image:\n')
    array_act, sorted_indices_act = torch.sort(prototype_activations[0])


    # Create a list of the identitities of the top-K most activated prototypes
    topk_proto_cls_ident = list()
    
    # Create a list of the prototypes (images)
    prototype_images = list()

    # Create a list of the prototypes (images) w/ bbox
    prototype_images_bboxes = list()

    # Create a list for prototypes with just one box
    prototype_images_just_this_box = list()

    # Create a list for prototypes patches list
    prototypes_patches_lists = list()

    # Create a list for the original images with all bboxes
    original_images_with_bboxes = list()

    # Create a list for the high activated patches
    high_activated_patches = list()

    # Create a list for the high activated patches in original image
    high_activated_patches_in_orig_image = list()

    # Create a list for the overlayed images
    overlayed_images = list()


    # Iterate through the most K activated prototypes
    for i in range(1, most_k_activated + 1):

        # Log into the report
        report.write(f'Top-{i} most activated prototype for this image:\n')


        # Save prototype
        prototype_fname = os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype.png' % i)
        report.write(f"Prototype: {prototype_fname}\n")
        prototype_img = save_prototype(
            fname=prototype_fname,
            load_img_dir=saved_prototypes_dir,
            index=sorted_indices_act[-i].item(),
            save_img=save_image
        )

        # Append to the corresponding list
        prototype_images.append(prototype_img)


        # Save prototype w/ bbox
        prototype_w_bbox_fname = os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype_with_box.png' % i)
        report.write(f"Prototype with bbox: {prototype_w_bbox_fname}\n")
        prototype_img_bbox = save_prototype_box(
            fname=prototype_w_bbox_fname,
            load_img_dir=saved_prototypes_dir,
            index=sorted_indices_act[-i].item()
        )

        # Append to the corresponding list
        prototype_images_bboxes.append(prototype_img_bbox)
        

        # Add some important information to the report
        report.write('Prototype index: {0}\n'.format(sorted_indices_act[-i].item()))
        report.write('Prototype class identity: {0}\n'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
        if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
            report.write('Prototype connection identity: {0}\n'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
        report.write('Activation value (similarity score): {0}\n'.format(array_act[-i]))
        report.write('Last layer connection with predicted class: {0}\n'.format(ppnet_model.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))


        # Append it to the list
        topk_proto_cls_ident.append(prototype_img_identity[sorted_indices_act[-i].item()])
        

        # Get activation pattern and upsampled activation pattern
        activation_pattern = prototype_activation_patterns[0][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
        
        # Save deformable-prototype information
        prototypes_just_this_box, prototypes_patches, original_img_j_with_boxes = save_deform_info(
            model=ppnet_model,
            offsets=offsets,
            input=original_img,
            activations=activation_pattern,
            save_dir=os.path.join(save_analysis_path, 'most_activated_prototypes'),
            prototype_img_filename_prefix='top-%d_activated_prototype_' % i,
            proto_index=sorted_indices_act[-i].item(),
            prototype_layer_stride=1,
            save_img=save_image
        )

        # Append to the corresponding lists
        prototype_images_just_this_box.append(prototypes_just_this_box)
        prototypes_patches_lists.append(prototypes_patches)
        original_images_with_bboxes.append(original_img_j_with_boxes)

        # Show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
        report.write('Most highly activated patch of the chosen image by this prototype:\n')
        
        # Append to the corresponding list
        high_activated_patches.append(high_act_patch)
        
        # Save image (if needed)
        if save_image:
            plt.imsave(
                os.path.join(save_analysis_path, 'most_activated_prototypes', 'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                high_act_patch
            )


        report.write('Most highly activated patch by this prototype shown in the original image:\n')
        high_activated_patch_in_orig_img = imsave_with_bbox(
            fname=os.path.join(save_analysis_path, 'most_activated_prototypes', 'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
            img_rgb=original_img,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3],
            color=(0, 255, 255),
            save_img=save_image
        )

        # Append it to the corresponding list
        high_activated_patches_in_orig_image.append(high_activated_patch_in_orig_img)


        # Show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        report.write('Prototype activation map of the chosen image:\n')

        # Append it to the corresponding list
        overlayed_images.append(overlayed_img)

        # Save image (if necessary)
        if save_image:
            # plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes', 'prototype_activation_map_by_top-%d_prototype.png' % i), overlayed_img)
        # log('--------------------------------------------------------------')
        report.write("\n")
    

    # Close report
    report.close()


    return test_image_name, correct_cls, predicted_cls, nr_prototypes_cls_ident, topk_proto_cls_ident, prototype_images, prototype_images_bboxes, prototype_images_just_this_box, prototypes_patches_lists, original_images_with_bboxes, high_activated_patches, high_activated_patches_in_orig_image, overlayed_images



# FIXME: Function: Get prototypes from top-K classes
def get_prototypes_from_topk_classes(k, logits, idx, ppnet_model, save_analysis_path, saved_prototypes_dir, prototype_activations, prototype_img_identity, prototype_max_connection, prototype_activation_patterns, img_size, original_img, predicted_cls, correct_cls, offsets):
    
    # Get prototypes from top-K classes
    # k = 2
    print('Prototypes from top-%d classes:' % k)


    # Get sorted indices activations
    _, sorted_indices_act = torch.sort(prototype_activations[idx])

    topk_logits, topk_classes = torch.topk(logits[idx], k=k)
    for i,c in enumerate(topk_classes.detach().cpu().numpy()):
        if not os.path.isdir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1))):
            os.makedirs(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))


        print('Top %d predicted class: %d' % (i+1, c))
        print('Logit of the class: %f' % topk_logits[i])
        class_prototype_indices = np.nonzero(ppnet_model.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_cnt = 1
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]


            # TODO: Erase uppon review
            # save_prototype_box(
            #     fname=None, #FIXME
            #     load_img_dir=os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype_with_box.png' % (i+1)),
            #     index=sorted_indices_act[-i].item()
            # )
            prototype_w_bbox_fname = os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype_with_box.png' % i)
            save_prototype_box(
                fname=prototype_w_bbox_fname,
                load_img_dir=saved_prototypes_dir,
                index=sorted_indices_act[-i].item()
            )
            
            
            print('Prototype index: {0}'.format(prototype_index))
            print('Prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
            if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                print('Prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
            print('Activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
            print('Last layer connection: {0}'.format(ppnet_model.last_layer.weight[c][prototype_index]))
            

            # Get activation pattern and upsampled activation pattern
            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)


            # Save information related to the deformable prototype(s)
            save_deform_info(
                model=ppnet_model,
                offsets=offsets,
                input=original_img, activations=activation_pattern,
                save_dir=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)),
                prototype_img_filename_prefix='top-%d_activated_prototype_' % prototype_cnt,
                proto_index=prototype_index,
                prototype_layer_stride=1
            )
            
            # Show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
            print('Most highly activated patch of the chosen image by this prototype:')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt), high_act_patch)
            print('Most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(
                fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
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
            print('prototype activation map of the chosen image:')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt), overlayed_img)

            # Update prototype counter
            prototype_cnt += 1


    # Log the result of predictions
    if predicted_cls == correct_cls:
        print('Prediction is correct.')
    else:
        print('Prediction is wrong.')


    return
