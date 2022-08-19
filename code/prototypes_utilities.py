# Sources: https://github.com/cfchen-duke/ProtoPNet/blob/master/push.py and https://github.com/cfchen-duke/ProtoPNet/blob/master/receptive_field.py
# Imports
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# PyTorch Imports
import torch



# Function: (Helper) Find high activation crop
def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0

    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break

    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break

    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break

    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break


    return lower_y, upper_y+1, lower_x, upper_x+1



# Function: Compute layer RF info
def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding, previous_layer_rf_info):
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]



# Function: Compute RF proto layer at spatial location
def compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info):
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert(height_index < n)
    assert(width_index < n)

    center_h = start + (height_index*j)
    center_w = start + (width_index*j)

    rf_start_height_index = max(int(center_h - (r/2)), 0)
    rf_end_height_index = min(int(center_h + (r/2)), img_size)

    rf_start_width_index = max(int(center_w - (r/2)), 0)
    rf_end_width_index = min(int(center_w + (r/2)), img_size)

    return [rf_start_height_index, rf_end_height_index,
            rf_start_width_index, rf_end_width_index]



# Function: Compute RF prototype
def compute_rf_prototype(img_size, prototype_patch_index, protoL_rf_info):
    img_index = prototype_patch_index[0]
    height_index = prototype_patch_index[1]
    width_index = prototype_patch_index[2]
    rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                       height_index,
                                                       width_index,
                                                       protoL_rf_info)
    return [img_index, rf_indices[0], rf_indices[1],
            rf_indices[2], rf_indices[3]]



# Function: Compute RF prototypes
def compute_rf_prototypes(img_size, prototype_patch_indices, protoL_rf_info):
    rf_prototypes = []
    for prototype_patch_index in prototype_patch_indices:
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                           height_index,
                                                           width_index,
                                                           protoL_rf_info)
        rf_prototypes.append([img_index, rf_indices[0], rf_indices[1],
                              rf_indices[2], rf_indices[3]])
    return rf_prototypes



# Function: Compute proto layer RF info
def compute_proto_layer_rf_info(img_size, cfg, prototype_kernel_size):
    rf_info = [img_size, 1, 1, 0.5]

    for v in cfg:
        if v == 'M':
            rf_info = compute_layer_rf_info(layer_filter_size=2,
                                            layer_stride=2,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)
        else:
            rf_info = compute_layer_rf_info(layer_filter_size=3,
                                            layer_stride=1,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info



# Function: Compute proto layer RF info (Version 2)
def compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):

    assert(len(layer_filter_sizes) == len(layer_strides))
    assert(len(layer_filter_sizes) == len(layer_paddings))

    rf_info = [img_size, 1, 1, 0.5]

    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]

        rf_info = compute_layer_rf_info(layer_filter_size=filter_size,
                                layer_stride=stride_size,
                                layer_padding=padding_size,
                                previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info



# Function: Push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    prototype_activation_function_in_numpy=None,
                    device='cpu'):


    # Put model into evaluation mode
    prototype_network_parallel.eval()
    print('Push Phase')

    # Start time
    start = time.time()
    # prototype_shape = prototype_network_parallel.module.prototype_shape
    prototype_shape = prototype_network_parallel.prototype_shape
    # n_prototypes = prototype_network_parallel.module.num_prototypes
    n_prototypes = prototype_network_parallel.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, 'epoch-'+str(epoch_number))
            
            # TODO: Erase uppon review
            # makedir(proto_epoch_dir)
            if not os.path.exists(proto_epoch_dir):
                os.makedirs(proto_epoch_dir)
        
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    # num_classes = prototype_network_parallel.module.num_classes
    num_classes = prototype_network_parallel.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                                   device=device)


    # TODO: Keep upon review
    # We don't need to strict this to the epoch
    # if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
    if proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'), proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'), proto_bound_boxes)

    print('Executing push...')
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    # prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(device))
    # prototype_network_parallel.cuda()
    # end = time.time()
    print('Push time: \t{0}'.format(time.time() -  start))



# Function: Update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               device='cpu'):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        # search_batch = search_batch.cuda()
        search_batch = search_batch.to(device)
        # this computation currently is not parallelized
        # protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    # prototype_shape = prototype_network_parallel.module.prototype_shape
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            # target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            target_class = torch.argmax(prototype_network_parallel.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch, :, fmap_height_start_index:fmap_height_end_index, fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            # protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            protoL_rf_info = prototype_network_parallel.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2], rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            # if prototype_network_parallel.module.prototype_activation_function == 'log':
            if prototype_network_parallel.prototype_activation_function == 'log':
                # proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.epsilon))
            
            # elif prototype_network_parallel.module.prototype_activation_function == 'linear':
            elif prototype_network_parallel.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size), interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1], proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes, prototype_self_act_filename_prefix + str(j) + '.npy'), proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    
                    # Save the whole image containing the prototype as png
                    # TODO: Erase uppon review
                    # plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original' + str(j) + '.png'), original_img_j, vmin=0.0, vmax=1.0)
                    pil_original_img_j = (original_img_j.copy() * 255).astype(np.uint8)
                    pil_original_img_j = Image.fromarray(pil_original_img_j.copy()).convert('RGB')
                    pil_original_img_j.save(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original' + str(j) + '.png'))
                    
                    # Overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    # TODO: Erase uppon review
                    # plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'), overlayed_original_img_j, vmin=0.0, vmax=1.0)
                    pil_overlayed_original_img_j = (overlayed_original_img_j.copy() * 255).astype(np.uint8)
                    pil_overlayed_original_img_j = Image.fromarray(pil_overlayed_original_img_j.copy()).convert('RGB')
                    pil_overlayed_original_img_j.save(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'))
                    
                    # If different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        # TODO: Erase uppon review
                        # plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'), rf_img_j, vmin=0.0, vmax=1.0)
                        pil_rf_img_j = (rf_img_j.copy() * 255).astype(np.uint8)
                        pil_rf_img_j = Image.fromarray(pil_rf_img_j.copy()).convert('RGB')
                        pil_rf_img_j.save(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'))
                        

                        # TODO: Erase uppon review
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2], rf_prototype_j[3]:rf_prototype_j[4]]
                        # plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'), overlayed_rf_img_j, vmin=0.0, vmax=1.0)
                        pil_overlayed_rf_img_j = (overlayed_rf_img_j.copy() * 255).astype(np.uint8)
                        pil_overlayed_rf_img_j = Image.fromarray(pil_overlayed_rf_img_j.copy()).convert('RGB')
                        pil_overlayed_rf_img_j.save(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'))
                    

                    # Save the prototype image (highly activated region of the whole image)
                    # TODO: Erase uppon review
                    # plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + str(j) + '.png'), proto_img_j, vmin=0.0, vmax=1.0)
                    pil_proto_img_j = (proto_img_j.copy() * 255).astype(np.uint8)
                    pil_proto_img_j = Image.fromarray(pil_proto_img_j.copy()).convert('RGB')
                    pil_proto_img_j.save(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + str(j) + '.png'))

    if class_specific:
        del class_to_img_index_dict
