# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/local_analysis.py
# Note: Finding the nearest prototypes to a test image

# Imports
import os
import argparse
import pandas as pd

# PyTorch Imports
import torch
import torchvision
import torch.utils.data

# Project Imports
from data_utilities import save_preprocessed_img, save_prototype, save_prototype_self_activation, save_prototype_original_img_with_bbox, imsave_with_bbox, CUB2002011Dataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
from prototypes_retrieval_utilities import retrieve_image_prototypes, get_prototypes_from_topk_classes
from prototypes_utilities import find_high_activation_crop
from train_val_test_utilities import model_test



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PH2, STANFORDCARS.")

# Model
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, resnet18, vgg19.')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation.")

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms.")

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")

# Get checkpoint
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint that contains weights and model parameters.")

# Compute metrics on test
parser.add_argument("--compute_metrics", action="store_true", help="Compute metrics on a specific data subset.")



# Parse the arguments
args = parser.parse_args()


# Get checkpoint (that contains the weights)
CHECKPOINT = args.checkpoint

# Data directory
DATA_DIR = args.data_dir

# Dataset
DATASET = args.dataset

# Base Architecture
BASE_ARCHITECTURE = args.base_architecture

# Results Directory
OUTPUT_DIR = args.output_dir

# Number of workers (threads)
WORKERS = args.num_workers

# Prototype activation function
PROTOTYPE_ACTIVATION_FUNCTION = args.prototype_activation_function

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.img_size

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type

# Compute metrics on data subset
COMPUTE_METRICS = args.compute_metrics



# Get the directory of results
results_dir = os.path.join(OUTPUT_DIR, CHECKPOINT)


# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



# Test Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


# Dataset
# CUB2002011
if DATASET == "CUB2002011":

    # Get image directories
    data_path = os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped")
    image_directories = [f for f in os.listdir(data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = CUB2002011Dataset(
        data_path=data_path,
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(test_set.labels_dict)

    # Labels dictionary
    labels_dict = test_set.labels_dict.copy()


# PH2
elif DATASET == "PH2":
    # Test Dataset
    test_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        augmented=False,
        transform=test_transforms
    )

    # Number of Classes
    NUM_CLASSES = len(test_set.diagnosis_dict)


# STANFORDCARS
elif DATASET == "STANFORDCARS":

    # Get image directories
    data_path = os.path.join(DATA_DIR, "stanfordcars", "cars_test", "images_cropped")
    image_directories = [f for f in os.listdir(data_path) if not f.startswith('.')]

    # Test Dataset
    test_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_test",
        augmented=False,
        cropped=True,
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(test_set.class_names)

    # Labels dictionary
    labels_dict = test_set.labels_dict.copy()



# Define prototype shape according to original paper
# The number of prototypes can be chosen with prior domain knowledge or hyperparameter search: we used 10 prototypes per class
NUM_PROTOTYPES_CLASS = int(NUM_CLASSES * 10)

# For VGG-16, VGG-19, DenseNet-121, DenseNet-161, we used 128 as the number of channels in a prototype
if BASE_ARCHITECTURE.lower() in ("densenet121", "densenet161", "vgg16", "vgg19"):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 128, 1, 1)

# For ResNet-34, we used 256 as the number of channels in a prototype;
elif BASE_ARCHITECTURE.lower() in ("resnet34"):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 256, 1, 1)

# For ResNet-152, we used 512 as the number of channels in a prototype
elif BASE_ARCHITECTURE.lower() in ("resnet152"):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 512, 1, 1)



# Create Test DataLoader
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)


# Weights
weights_dir = os.path.join(results_dir, "weights")

# Prototypes
load_img_dir = os.path.join(weights_dir, 'prototypes')


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Construct the Model
ppnet_model = construct_PPNet(
    base_architecture=BASE_ARCHITECTURE,
    pretrained=False,
    img_size=IMG_SIZE,
    prototype_shape=PROTOTYPE_SHAPE,
    num_classes=NUM_CLASSES,
    prototype_activation_function=PROTOTYPE_ACTIVATION_FUNCTION,
    add_on_layers_type=ADD_ON_LAYERS_TYPE
)


# Define if the model should be class specific
class_specific = True



# Put model into DEVICE (CPU or GPU)
ppnet_model = ppnet_model.to(DEVICE)

# TODO: Decide which of model weights should we load
# model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
# model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
model_weights = torch.load(model_path, map_location=DEVICE)
ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
print("Model weights loaded with success.")


# Create a local analysis path
save_analysis_path = os.path.join(results_dir, "analysis", "local")
if not os.path.isdir(save_analysis_path):
    os.makedirs(save_analysis_path)


# Get prototype shape
prototype_shape = ppnet_model.prototype_shape

# Get max distance
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]


# Get model performance metrics
# accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=print)
if COMPUTE_METRICS:
    metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
    test_accuracy = metrics_dict["accuracy"]
    print(f"Accuracy on test: {test_accuracy}.")



# Analysis .CSV
analysis_dict = {
    "Image Filename":list(),
    "Ground-Truth Label":list(),
    "Predicted Label":list(),
    "Number Prototypes Connected Class Identity":list(),
    "Top-10 Activated Prototypes":list()
}


# Go through all image directories
for image_dir in image_directories:

    # Get images in this directory
    image_names = [i for i in os.listdir(os.path.join(data_path, image_dir)) if not i.startswith('.')]
    image_names = [i for i in image_names if not os.path.isdir(os.path.join(data_path, i))]

    # Go through all images in a single directory
    for image_name in image_names:

        # Get image label
        image_label = labels_dict[image_dir]

        # Create image analysis path
        image_analysis_path = os.path.join(save_analysis_path, image_dir, image_name.split('.')[0])
        if not os.path.isdir(image_analysis_path):
            os.makedirs(image_analysis_path)

        # Analyse this image
        img_fname, gt_label, pred_label, nr_prototypes_cls_ident, topk_proto_cls_ident = retrieve_image_prototypes(
            save_analysis_path=image_analysis_path,
            weights_dir=weights_dir,
            load_img_dir=load_img_dir,
            ppnet_model=ppnet_model,
            device=DEVICE,
            test_transforms=test_transforms,
            test_image_dir=os.path.join(data_path, image_dir),
            test_image_name=image_name,
            test_image_label=image_label,
            norm_params={"mean":MEAN, "std":STD},
            img_size=IMG_SIZE
        )


        # Add information to our data dictionary
        analysis_dict["Image Filename"].append(img_fname)
        analysis_dict["Ground-Truth Label"].append(gt_label)
        analysis_dict["Predicted Label"].append(pred_label)
        analysis_dict["Number Prototypes Connected Class Identity"].append(nr_prototypes_cls_ident)
        analysis_dict["Top-10 Activated Prototypes"].append(topk_proto_cls_ident)


# Save data dictionary into a .CSV
analysis_df = pd.DataFrame.from_dict(data=analysis_dict)
analysis_df.to_csv(path_or_buf=os.path.join(save_analysis_path, "analysis.csv"))


# parser.add_argument('-imgdir', nargs=1, type=str)
# parser.add_argument('-img', nargs=1, type=str)
# parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
# args = parser.parse_args()


# # specify the test image to be analyzed
# test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
# test_image_name = args.img[0] #'Painted_Bunting_0081_15230.jpg'
# test_image_label = args.imgclass[0] #15

# test_image_path = os.path.join(test_image_dir, test_image_name)



# # Sanity check: Confirm prototype class identity
# # TODO: load_img_dir = os.path.join(load_model_dir, 'img')
# saved_prototypes_dir = os.path.join(weights_dir, 'prototypes')

# # TODO: prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
# prototype_info = np.load(os.path.join(saved_prototypes_dir, 'bb.npy'))
# prototype_img_identity = prototype_info[:, -1]

# print('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
# print('Their class identities are: ' + str(prototype_img_identity))


# # Confirm prototype connects most strongly to its own class
# prototype_max_connection = torch.argmax(ppnet_model.last_layer.weight, dim=0)
# prototype_max_connection = prototype_max_connection.cpu().numpy()
# if np.sum(prototype_max_connection == prototype_img_identity) == ppnet_model.num_prototypes:
#     print('All prototypes connect most strongly to their respective classes.')
# else:
#     print('WARNING: Not all prototypes connect most strongly to their respective classes.')



# # Load the image and labels
# img_pil = Image.open(test_image_path)
# img_tensor = test_transforms(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
# images_test = img_variable.to(DEVICE)
# labels_test = torch.tensor([test_image_label])

# # Run inference with ppnet
# logits, min_distances = ppnet_model(images_test)
# conv_output, distances = ppnet_model.push_forward(images_test)
# prototype_activations = ppnet_model.distance_2_similarity(min_distances)
# prototype_activation_patterns = ppnet_model.distance_2_similarity(distances)
# if ppnet_model.prototype_activation_function == 'linear':
#     prototype_activations = prototype_activations + max_dist
#     prototype_activation_patterns = prototype_activation_patterns + max_dist


# # Create tables of predictions and ground-truth
# tables = []
# for i in range(logits.size(0)):
#     tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
#     print(str(i) + ' ' + str(tables[-1]))

# idx = 0
# predicted_cls = tables[idx][0]
# correct_cls = tables[idx][1]
# print('Predicted: ' + str(predicted_cls))
# print('Actual: ' + str(correct_cls))
# original_img = save_preprocessed_img(
#     fname=os.path.join(save_analysis_path, 'original_img.png'),
#     preprocessed_imgs=images_test,
#     mean=MEAN, 
#     std=STD, 
#     index=idx
# )



# # Get most activated (nearest) 10 prototypes of this image
# # TODO: makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))
# if not os.path.isdir(os.path.join(save_analysis_path, 'most_activated_prototypes')):
#     os.makedirs(os.path.join(save_analysis_path, 'most_activated_prototypes'))

# print('Most activated 10 prototypes of this image:')
# array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
# for i in range(1, 11):
#     print('top {0} activated prototype for this image:'.format(i))
    
#     # Save prototype
#     save_prototype(
#         fname=os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype.png' % i),
#         load_img_dir=saved_prototypes_dir,
#         epoch=None, 
#         index=sorted_indices_act[-i].item()
#     )


#     # Save prototype original image with bounding-box
#     save_prototype_original_img_with_bbox(
#         fname=os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype_in_original_pimg.png' % i),
#         epoch=None,
#         index=sorted_indices_act[-i].item(),
#         bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
#         bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
#         bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
#         bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
#         color=(0, 255, 255)
#     )


#     # Save prototype self-activation
#     save_prototype_self_activation(
#         fname=os.path.join(save_analysis_path, 'most_activated_prototypes', 'top-%d_activated_prototype_self_act.png' % i),
#         load_img_dir=saved_prototypes_dir,
#         epoch=None,
#         index=sorted_indices_act[-i].item()
#     )
    
#     print('prototype index: {0}'.format(sorted_indices_act[-i].item()))
#     print('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    
#     if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
#         print('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    
#     print('activation value (similarity score): {0}'.format(array_act[-i]))
#     print('last layer connection with predicted class: {0}'.format(ppnet_model.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
    
#     activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
#     upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

#     # show the most highly activated patch of the image by this prototype
#     high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
#     high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
#     print ('most highly activated patch of the chosen image by this prototype:')
#     # plt.axis('off')
#     plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes', 'most_highly_activated_patch_by_top-%d_prototype.png' % i), high_act_patch)

#     print('most highly activated patch by this prototype shown in the original image:')
#     imsave_with_bbox(
#         fname=os.path.join(save_analysis_path, 'most_activated_prototypes', 'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
#         img_rgb=original_img,
#         bbox_height_start=high_act_patch_indices[0],
#         bbox_height_end=high_act_patch_indices[1],
#         bbox_width_start=high_act_patch_indices[2],
#         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255)
#     )

#     # show the image overlayed with prototype activation map
#     rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
#     rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
#     heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[...,::-1]
#     overlayed_img = 0.5 * original_img + 0.3 * heatmap
#     print('prototype activation map of the chosen image:')
#     # plt.axis('off')
#     plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes', 'prototype_activation_map_by_top-%d_prototype.png' % i), overlayed_img)
#     print('--------------------------------------------------------------')



# # Get prototypes from top-K classes
# k = 50
# print('Prototypes from top-%d classes:' % k)
# topk_logits, topk_classes = torch.topk(logits[idx], k=k)
# for i, c in enumerate(topk_classes.detach().cpu().numpy()):
#     # TODO: makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))
#     if not os.path.isdir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1))):
#         os.makedirs(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

#     print('top %d predicted class: %d' % (i+1, c))
#     print('logit of the class: %f' % topk_logits[i])
#     class_prototype_indices = np.nonzero(ppnet_model.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
#     class_prototype_activations = prototype_activations[idx][class_prototype_indices]
#     _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

#     prototype_cnt = 1
#     for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
#         prototype_index = class_prototype_indices[j]

#         # Save prototype
#         save_prototype(
#             fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_activated_prototype.png' % prototype_cnt),
#             load_img_dir=saved_prototypes_dir,
#             epoch=None,
#             index=prototype_index
#         )
    

#         # Save prototype original image with bounding-box
#         save_prototype_original_img_with_bbox(
#             fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
#             load_img_dir=saved_prototypes_dir,
#             epoch=None,
#             index=prototype_index,
#             bbox_height_start=prototype_info[prototype_index][1],
#             bbox_height_end=prototype_info[prototype_index][2],
#             bbox_width_start=prototype_info[prototype_index][3],
#             bbox_width_end=prototype_info[prototype_index][4],
#             color=(0, 255, 255)
#         )


#         # Save prototype self-activation
#         save_prototype_self_activation(
#             fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'top-%d_activated_prototype_self_act.png' % prototype_cnt),
#             load_img_dir=saved_prototypes_dir,
#             epoch=None,
#             index=prototype_index
#         )


#         print('prototype index: {0}'.format(prototype_index))
#         print('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
        
#         if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
#             print('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
        
#         print('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
#         print('last layer connection: {0}'.format(ppnet_model.last_layer.weight[c][prototype_index]))
        
#         activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
#         upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
#         # show the most highly activated patch of the image by this prototype
#         high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
#         high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :]
#         print('most highly activated patch of the chosen image by this prototype:')
#         # plt.axis('off')
#         plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt), high_act_patch)
#         print('most highly activated patch by this prototype shown in the original image:')
        
#         # Save image with bounding-box
#         imsave_with_bbox(
#             fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
#             img_rgb=original_img,
#             bbox_height_start=high_act_patch_indices[0],
#             bbox_height_end=high_act_patch_indices[1],
#             bbox_width_start=high_act_patch_indices[2],
#             bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255)
#         )
        
#         # show the image overlayed with prototype activation map
#         rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
#         rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
#         heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
#         heatmap = np.float32(heatmap) / 255
#         heatmap = heatmap[...,::-1]
#         overlayed_img = 0.5 * original_img + 0.3 * heatmap
#         print('prototype activation map of the chosen image:')
#         # plt.axis('off')
#         plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1), 'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt), overlayed_img)
#         print('--------------------------------------------------------------')
#         prototype_cnt += 1
#     print('***************************************************************')


# # Check if predicted class is the correct class
# if predicted_cls == correct_cls:
#     print('Prediction is correct.')
# else:
#     print('Prediction is wrong.')



print("Finished.")
