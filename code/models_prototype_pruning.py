# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/run_pruning.py

# Imports
import os
import argparse
import numpy as np

# PyTorch Imports
import torch
import torch.utils.data
import torchvision

# Project Imports
from data_utilities import preprocess_input_function, CUB2002011Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
from prototypes_utilities import push_prototypes
from pruning_utilities import prune_prototypes
from train_val_test_utilities import model_train, model_test, last_only



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "STANFORDCARS"], help="Data set: CUB2002011, STANFORDCARS.")

# Model
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, resnet18, vgg19, ')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")

# Prototype shape
# prototype_shape = (2000, 128, 1, 1)
# parser.add_argument('--prototype_shape', type=tuple, default=(2000, 128, 1, 1), help="Prototype shape.")

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Class Weights
# optimize_last_layer = True
parser.add_argument("--optimize_last_layer", action="store_true", default=True, help="Optimize last layer.")

# Last layer optimizer learning rate
# last_layer_optimizer_lr = 1e-4
parser.add_argument('--last_layer_optimizer_lr', type=float, default=1e-4, help="Last layer optimizer learning rate.")

# Loss coeficients
# coefs = {'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}
# coefs = {'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}
parser.add_argument('--coefs', type=dict, default={'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}, help="Loss coeficients.")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Checkpoint
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from which to resume training")

# Pruning paramters
# k = 6
parser.add_argument("--k", type=int, default=6, help="K.")

# Pruning threshold
# prune_threshold = 3
parser.add_argument("--prune_threshold", type=int, default=3, help="Prune threshold.")



# Parse the arguments
args = parser.parse_args()


# Training checkpoint
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

# LAST_LAYER_OPTIMIZER_LR
LAST_LAYER_OPTIMIZER_LR = args.last_layer_optimizer_lr

# COEFS (weighting of different training losses)
COEFS = args.coefs

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.img_size

# Prototype shape
PROTOTYPE_SHAPE = args.prototype_shape

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type

# Optimized last later
OPTMIZE_LAST_LAYER = args.optimize_last_layer

# K
K = args.k

# Prune threshold
PRUNE_THRESHOLD = args.prune_threshold


# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_height = IMG_SIZE
img_width = IMG_SIZE



# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Push Transforms
train_push_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor()
])

# Test Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


# Dataset
# CUB2002011
if DATASET == "CUB2002011":
    # Train Dataset
    train_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        transform=train_transforms
    )

    # Train Push Dataset (Prototypes)
    train_push_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
         transform=train_push_transforms
    )

    # Test Dataset
    test_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.labels_dict)



# STANFORDCARS
elif DATASET == "STANFORDCARS":
    # Train Dataset
    train_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_train",
        cropped=True,
        transform=train_transforms
    )

    # Train Push Dataset (Prototypes)
    train_push_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_train",
        cropped=True,
        transform=train_push_transforms
    )

    # Test Dataset
    test_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_test",
        cropped=True,
        transform=test_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.class_names)



# Get the directory of results
results_dir = os.path.join(OUTPUT_DIR, CHECKPOINT)

# Results and Weights
weights_dir = os.path.join(results_dir, "weights")

# Pruned model dir
pruned_model_dir = os.path.join(weights_dir, f'pruned_prototypes_k{K}_pt{PRUNE_THRESHOLD}')

if not os.path.isdir(pruned_model_dir):
    os.makedirs(pruned_model_dir)

# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")



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



# TODO: Erase uppon review
# parser.add_argument('-modeldir', nargs=1, type=str)
# parser.add_argument('-model', nargs=1, type=str)

# original_model_dir = args.modeldir[0] #'./saved_models/densenet161/003/'
# original_model_name = args.model[0] #'10_16push0.8007.pth'

# need_push = ('nopush' in original_model_name)
# if need_push:
#     assert(False) # pruning must happen after push
# else:
#     epoch = original_model_name.split('push')[0]

# if '_' in epoch:
#     epoch = int(epoch.split('_')[0])
# else:
#     epoch = int(epoch)

# pruned_model_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch, k, prune_threshold))
# if not os.path.isdir(model_dir):
#     os.makedirs(model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)



# Construct the Model
ppnet_model = construct_PPNet(
    base_architecture=BASE_ARCHITECTURE,
    pretrained=True,
    img_size=IMG_SIZE,
    prototype_shape=PROTOTYPE_SHAPE,
    num_classes=NUM_CLASSES,
    prototype_activation_function=PROTOTYPE_ACTIVATION_FUNCTION,
    add_on_layers_type=ADD_ON_LAYERS_TYPE)

# if prototype_activation_function == 'linear':
#     ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)


# Define if we want the model to be class specific
class_specific = True


# Put model into DEVICE (CPU or GPU)
ppnet_model = ppnet_model.to(DEVICE)


# Load model weights
model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
model_weights = torch.load(model_path, map_location=DEVICE)
ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
print("Model weights loaded with success.")


# TODO: Erase uppon review
# load the data
# from settings import train_dir, test_dir, train_push_dir
# train_batch_size = 80
# test_batch_size = 100
# img_size = 224
# train_push_batch_size = 80


# DataLoaders
# Train Loader
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=False)

# Train Push Loader ( The push set is needed for pruning because it is unnormalized)
train_push_loader = torch.utils.data.DataLoader(dataset=train_push_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=False)

# Test Loader
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=False)



# Log prints
print('Train set size: {0}'.format(len(train_loader.dataset)))
print('Test set size: {0}'.format(len(test_loader.dataset)))
print('Train Push set size: {0}'.format(len(train_push_loader.dataset)))
print('Batch size: {0}'.format(BATCH_SIZE))


# Define some extra paths and directories
# Directory for saved prototypes
saved_prototypes_dir = os.path.join(weights_dir, 'prototypes')
if not os.path.isdir(saved_prototypes_dir):
    os.makedirs(saved_prototypes_dir)


# Output weight matrix filename
weight_matrix_filename = 'outputL_weights'

# Prefix for prototype images
prototype_img_filename_prefix = 'prototype-img'

# Prefix for prototype self activation
prototype_self_act_filename_prefix = 'prototype-self-act'

# Prefix for prototype bouding boxes
proto_bound_boxes_filename_prefix = 'bb'





# Best Accuracy Variable
best_accuracy = -np.inf

# TODO
# tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
test_accuracy = metrics_dict["accuracy"]
print(f"Accuracy on the test set, before pruning: {test_accuracy}")

# Update best accuracy
if test_accuracy > best_accuracy:
    best_accuracy = test_accuracy



# We must execute the push operation first
print("Pushing prototypes...")
push_prototypes(
    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
    prototype_network_parallel=ppnet_model, # pytorch network with prototype_vectors
    class_specific=class_specific,
    preprocess_input_function=preprocess_input_function, # normalize if needed (FIXME: according to original implementation)
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=saved_prototypes_dir, # if not None, prototypes will be saved here
    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
    prototype_img_filename_prefix=prototype_img_filename_prefix,
    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
    save_prototype_class_identity=True,
    device=DEVICE
)

metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
test_accuracy = metrics_dict["accuracy"]
print(f"Accuracy on the test set, after pushing: {test_accuracy}")

# Update best accuracy
if test_accuracy > best_accuracy:
    best_accuracy = test_accuracy



# Prune prototypes
print('Pruning prototypes...')
prune_prototypes(
    dataloader=train_push_loader,
    prototype_network_parallel=ppnet_model,
    device=DEVICE,
    k=K,
    prune_threshold=PRUNE_THRESHOLD,
    preprocess_input_function=preprocess_input_function, # normalize
    original_model_dir=weights_dir,
    epoch_number=None,
    copy_prototype_imgs=True
)


# Get performance metrics
# accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
metrics_dict, _ = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
test_prune_accuracy = metrics_dict["accuracy"]
print(f"Accuracy on the test set, after pruning: {test_prune_accuracy}")

# Save new model
# save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=original_model_name.split('push')[0] + 'prune', accu=accu, target_accu=0.70, log=log)
if test_prune_accuracy > best_accuracy:
    print(f"Accuracy increased from {best_accuracy} to {test_prune_accuracy}. Saving new model...")

    # Model path
    model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_pruned.pt")
    save_dict = {
        'model_state_dict':ppnet_model.state_dict(),
    }

    torch.save(save_dict, model_path)
    print(f"Successfully saved at: {model_path}")

    # Update best accuracy value
    best_accuracy = test_prune_accuracy



# Last layer optimization
if OPTMIZE_LAST_LAYER:

    # Define optimizers and learning rate schedulers
    # Last Layer Optimizer
    last_layer_optimizer_specs = [

        {'params': ppnet_model.last_layer.parameters(), 'lr': LAST_LAYER_OPTIMIZER_LR}
        
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)


    # Begin last layer optimization
    print('Optimizing last layer...')
    last_only(model=ppnet_model)

    for i in range(100):
        print('iteration: \t{0}'.format(i))

        # Train Phase
        # _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, class_specific=class_specific, coefs=coefs, log=log)
        metrics_dict = model_train(model=ppnet_model, dataloader=train_loader, device=DEVICE, optimizer=last_layer_optimizer, class_specific=class_specific, coefs=coefs)

        # Test Phase
        # accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=log)
        metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
        test_prune_last_accuracy = metrics_dict["accuracy"]

        # Save new model
        # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune', accu=accu, target_accu=0.70, log=log)
        if test_prune_last_accuracy > best_accuracy:
            print(f"Accuracy increased from {best_accuracy} to {test_prune_last_accuracy}. Saving new model...")

            # Model path
            model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_pruned_last_k{K}_pt{PRUNE_THRESHOLD}.pt")
            save_dict = {
                'model_state_dict':ppnet_model.state_dict(),
            }

            torch.save(save_dict, model_path)
            print(f"Successfully saved at: {model_path}")

            # Update best accuracy value
            best_accuracy = test_prune_last_accuracy



print("Finished.")
