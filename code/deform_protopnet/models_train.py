# Imports
import os
import argparse
import numpy as np
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.utils.class_weight import compute_class_weight

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

# Weights and Biases (W&B) Imports
import wandb

# Log in to W&B Account
wandb.login()

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from data_utilities import preprocess_input_function, CUB2002011Dataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
from prototypes_utilities import push_prototypes
from train_val_test_utilities import model_train, model_validation, print_metrics, warm_only, warm_pre_offset, joint, last_only



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
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")

# Margin
parser.add_argument('--margin', type=float, default=None)

# Subtractive margin
# subtractive_margin = True
parser.add_argument('--subtractive_margin', action="store_true", default=True)

# Using deformable convolution
parser.add_argument('--using_deform', type=str, default=None)

# Top-K
parser.add_argument('--topk_k', type=int, default=None)

# Deformable Convolution Hidden Channels
parser.add_argument('--deformable_conv_hidden_channels', type=int, default=None)

# Number of Prototypes
parser.add_argument('--num_prototypes', type=int, default=None)

# Dilation
parser.add_argument('--dilation', type=float, default=2)

# Incorrect class connection
parser.add_argument('--incorrect_class_connection', type=float, default=0)

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Joint optimizer learning rates
# joint_optimizer_lrs = {'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3, 'conv_offset': 1e-4, 'joint_last_layer_lr': 1e-5}
parser.add_argument('--joint_optimizer_lrs', type=dict, default={'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3, 'conv_offset': 1e-4, 'joint_last_layer_lr': 1e-5}, help="Joint optimizer learning rates.")

# Joint learning rate step size
# joint_lr_step_size = 5
parser.add_argument('--joint_lr_step_size', type=int, default=5, help="Joint learning rate step size.")

# Warm optimizer learning rates
# warm_optimizer_lrs = {'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}
parser.add_argument('--warm_optimizer_lrs', type=dict, default={'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}, help="Warm optimizer learning rates.")

# Warm pre-offset optimizer learning rates
# warm_pre_offset_optimizer_lrs = {'add_on_layers': 3e-3, 'prototype_vectors': 3e-3, 'features': 1e-4}
parser.add_argument('--warm_pre_offset_optimizer_lrs', type=dict, default={'add_on_layers': 3e-3, 'prototype_vectors': 3e-3, 'features': 1e-4}, help="Warm pre-offset optimizer learning rates.")

# Warm pre-prototype optimizer learning rates
# warm_pre_prototype_optimizer_lrs = {'add_on_layers': 3e-3, 'conv_offset': 3e-3, 'features': 1e-4}
parser.add_argument('--warm_pre_prototype_optimizer_lrs', type=dict, default={'add_on_layers': 3e-3, 'conv_offset': 3e-3, 'features': 1e-4}, help="Warm pre-prototype optimizer learning rates")

# Last layer optimizer learning rate
# last_layer_optimizer_lr = 1e-4
parser.add_argument('--last_layer_optimizer_lr', type=float, default=1e-4, help="Last layer optimizer learning rate.")

# Last layer fixed
# last_layer_fixed = True
parser.add_argument('--last_layer_fixed', action="store_true", default=True)

# Loss coeficients
# coefs = {'crs_ent': 1, 'clst': -0.8, 'sep': 0.08, 'l1': 1e-2, 'offset_bias_l2': 8e-1, 'offset_weight_l2': 8e-1, 'orthogonality_loss': 0.1}
parser.add_argument('--coefs', type=dict, default={'crs_ent': 1, 'clst': -0.8, 'sep': 0.08, 'l1': 1e-2, 'offset_bias_l2': 8e-1, 'offset_weight_l2': 8e-1, 'orthogonality_loss': 0.1}, help="Loss coeficients.")

# Push start
# push_start = 20
parser.add_argument('--push_start', type=int, default=20, help="Push start.")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Number of training epochs
# num_train_epochs = 31
parser.add_argument('--num_train_epochs', type=int, default=300, help="Number of training epochs.")

# Number of warm epochs
# num_warm_epochs = 5
parser.add_argument('--num_warm_epochs', type=int, default=5, help="Number of warm epochs.")

# Number of secondary warm epochs
# num_secondary_warm_epochs = 5
parser.add_argument('--num_secondary_warm_epochs', type=int, default=5, help="Number of secondary warm epochs.")


# Learning rate
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Save frequency
parser.add_argument("--save_freq", type=int, default=10, help="Frequency (in number of epochs) to save the model")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from which to resume training")



# Parse the arguments
args = parser.parse_args()


# Resume training
RESUME = args.resume

# Training checkpoint
if RESUME:
    CHECKPOINT = args.checkpoint

    assert CHECKPOINT is not None, "Please specify the model checkpoint when resume is True"


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

# Number of training epochs
NUM_TRAIN_EPOCHS = args.num_train_epochs

# Number of warm epochs
NUM_WARM_EPOCHS = args.num_warm_epochs

# Number of secondary warm epochs
NUM_SECONDARY_WARM_EPOCHS = args.num_secondary_warm_epochs

# Push epochs
# push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
PUSH_EPOCHS = [i for i in range(NUM_TRAIN_EPOCHS) if i % 10 == 0]


# Learning rate
LEARNING_RATE = args.lr

# Prototype activation function
PROTOTYPE_ACTIVATION_FUNCTION = args.prototype_activation_function

# Joint optimizer learning rates
JOINT_OPTIMIZER_LRS = args.joint_optimizer_lrs

# JOINT_LR_STEP_SIZE
JOINT_LR_STEP_SIZE = args.joint_lr_step_size

# WARM_OPTIMIZER_LRS
WARM_OPTIMIZER_LRS = args.warm_optimizer_lrs

# WARM_PRE_OFFSET_OPTIMIZER_LRS
WARM_PRE_OFFSET_OPTIMIZER_LRS = args.warm_pre_offset_optimizer_lrs

# LAST_LAYER_OPTIMIZER_LR
LAST_LAYER_OPTIMIZER_LR = args.last_layer_optimizer_lr

# COEFS (weighting of different training losses)
COEFS = args.coefs

# PUSH_START
PUSH_START = args.push_start

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.img_size

# Add on layers type
ADD_ON_LAYERS_TYPE = args.add_on_layers_type

# Last Layer Fixed
LAST_LAYER_FIXED = args.last_layer_fixed

# Save frquency
SAVE_FREQ = args.save_freq

# Resize (data transforms)
RESIZE_OPT = args.resize

# Margin
MARGIN = args.margin

# Using subtractive margin
SUBTRACTIVE_MARGIN = args.subtractive_margin

# Using deformable convolution
USING_DEFORM = args.using_deform

# TOPK_K
TOPK_K = args.topk_k

# Deformable convolutional hidden channels
DEFORMABLE_CONV_HIDDEN_CHANNELS = args.deformable_conv_hidden_channels

# Number of Prototypes
NUM_PROTOTYPES = args.num_prototypes

# Dilation
DILATION = args.dilation

# Incorrect class connection
INCORRECT_CLASS_CONNECTION = args.incorrect_class_connection



# TODO: Erase uppon review
# Debug prints
# print("USING DEFORMATION: ", using_deform)
# print("Margin set to: ", m)
# print("last_layer_fixed set to: {}".format(last_layer_fixed))
# print("subtractive_margin set to: {}".format(subtractive_margin))
# print("topk_k set to: {}".format(topk_k))
# print("num_prototypes set to: {}".format(num_prototypes))
# print("incorrect_class_connection: {}".format(incorrect_class_connection))
# print("deformable_conv_hidden_channels: {}".format(deformable_conv_hidden_channels))


# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(OUTPUT_DIR, DATASET.lower(), "deformable-protopnet", BASE_ARCHITECTURE.lower(), timestamp)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# Save training parameters
with open(os.path.join(results_dir, "train_params.txt"), "w") as f:
    f.write(str(args))



# Set the W&B project
wandb.init(
    project="proto-counterfactuals", 
    name=timestamp,
    config={
        "architecture": f"deform-{BASE_ARCHITECTURE.lower()}",
        "dataset": DATASET.lower(),
        "train-epochs": NUM_TRAIN_EPOCHS,
    }
)



# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_height = IMG_SIZE
img_width = IMG_SIZE



# Train Transforms (online augmentation parameters are commented, but they are from the original paper)
train_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomAffine(degrees=(-25, 25), shear=15),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Push Transforms (without Normalize)
train_push_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor()
])

# Validation Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
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
        augmented=True,
        transform=train_transforms
    )

    # Train Push Dataset (Prototypes)
    train_push_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=train_push_transforms
    )

    # Validation Dataset
    val_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=val_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.labels_dict)



# PH2
elif DATASET == "PH2":
    # Train Dataset
    train_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="train",
        cropped=True,
        augmented=True,
        transform=train_transforms
    )

    # Train Push Dataset (Prototypes)
    train_push_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="train",
        cropped=True,
        augmented=False,
        transform=train_push_transforms
    )

    # Validation Dataset
    val_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        augmented=False,
        transform=val_transforms
    )

    # Number of Classes
    NUM_CLASSES = len(train_set.diagnosis_dict)



# STANFORDCARS
elif DATASET == "STANFORDCARS":
    # Train Dataset
    train_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_train",
        augmented=True,
        cropped=True,
        transform=train_transforms
    )

    # Train Push Dataset (Prototypes)
    train_push_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_train",
        augmented=False,
        cropped=True,
        transform=train_push_transforms
    )

    # Validation Dataset
    val_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_test",
        augmented=False,
        cropped=True,
        transform=val_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.class_names)



# Results and Weights
weights_dir = os.path.join(results_dir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(results_dir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard"), flush_secs=30)


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")



# Define the number of prototypes per class
if NUM_PROTOTYPES == -1:
    NUM_PROTOTYPES_CLASS = 1200

if BASE_ARCHITECTURE.lower() == 'resnet34':
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 512, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

elif BASE_ARCHITECTURE.lower() in ('resnet50', 'resnet152'):
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 2048, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

elif BASE_ARCHITECTURE.lower() == 'densenet121':
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 1024, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

elif BASE_ARCHITECTURE.lower() == 'densenet161':
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 2208, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'

else:
    PROTOTYPE_SHAPE = (NUM_PROTOTYPES_CLASS, 512, 2, 2)
    ADD_ON_LAYERS_TYPE = 'upsample'


# Debug print
print("Add on layers type: ", ADD_ON_LAYERS_TYPE)



# Construct the model
ppnet_model = construct_PPNet(
    base_architecture=BASE_ARCHITECTURE.lower(),
    pretrained=True,
    img_size=IMG_SIZE,
    prototype_shape=PROTOTYPE_SHAPE,
    num_classes=NUM_CLASSES,
    topk_k=TOPK_K,
    m=MARGIN,
    add_on_layers_type=ADD_ON_LAYERS_TYPE,
    using_deform=USING_DEFORM,
    incorrect_class_connection=INCORRECT_CLASS_CONNECTION,
    deformable_conv_hidden_channels=DEFORMABLE_CONV_HIDDEN_CHANNELS,
    prototype_dilation=2
)
class_specific = True



# Define optimizers
# Joint Optimizer Specifications
if BASE_ARCHITECTURE.lower() == 'resnet152' and DATASET == 'stanford_dogs':
    JOINT_OPTIMIZER_LRS['features'] = 1e-5

joint_optimizer_specs = [
    {'params': ppnet_model.features.parameters(), 'lr': JOINT_OPTIMIZER_LRS['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet_model.add_on_layers.parameters(), 'lr': JOINT_OPTIMIZER_LRS['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet_model.prototype_vectors, 'lr': JOINT_OPTIMIZER_LRS['prototype_vectors']},
    {'params': ppnet_model.conv_offset.parameters(), 'lr': JOINT_OPTIMIZER_LRS['conv_offset']},
    {'params': ppnet_model.last_layer.parameters(), 'lr': JOINT_OPTIMIZER_LRS['joint_last_layer_lr']}
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=JOINT_LR_STEP_SIZE, gamma=0.2)


# Warm Optimizer Specifications
warm_optimizer_specs = [
    {'params': ppnet_model.add_on_layers.parameters(), 'lr': WARM_OPTIMIZER_LRS['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet_model.prototype_vectors, 'lr': WARM_OPTIMIZER_LRS['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)


# Warm Pre-Offset Optimizer Specifications
if BASE_ARCHITECTURE.lower() == 'resnet152' and DATASET == 'stanford_dogs':
    WARM_PRE_OFFSET_OPTIMIZER_LRS['features'] = 1e-5

warm_pre_offset_optimizer_specs = [
    {'params': ppnet_model.add_on_layers.parameters(), 'lr': WARM_PRE_OFFSET_OPTIMIZER_LRS['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet_model.prototype_vectors, 'lr': WARM_PRE_OFFSET_OPTIMIZER_LRS['prototype_vectors']},
    {'params': ppnet_model.features.parameters(), 'lr': WARM_PRE_OFFSET_OPTIMIZER_LRS['features'], 'weight_decay': 1e-3},
]
warm_pre_offset_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs)


# Warm Learning Rate Scheduler
if DATASET == 'stanford_dogs':
    warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)
else:
    warm_lr_scheduler = None


# Last Layer Optimizer
last_layer_optimizer_specs = [{'params': ppnet_model.last_layer.parameters(), 'lr': LAST_LAYER_OPTIMIZER_LR}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)



# Put model into DEVICE (CPU or GPU)
ppnet_model = ppnet_model.to(DEVICE)


# Get model summary
try:
    model_summary = summary(ppnet_model, (1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)

except:
    model_summary = str(ppnet_model)


# Write into file
with open(os.path.join(results_dir, "model_summary.txt"), 'w') as f:
    f.write(str(model_summary))



# TODO: Review
# Class weights for loss
# if args.classweights:
#     classes = np.array(range(NUM_CLASSES))
#     cw = compute_class_weight('balanced', classes=classes, y=np.array(train_set.images_labels))
#     cw = torch.from_numpy(cw).float().to(DEVICE)
#     print(f"Using class weights {cw}")
# else:
#     cw = None




# Resume training from given checkpoint
if RESUME:
    checkpoint = torch.load(CHECKPOINT)
    ppnet_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    warm_optimizer.load_state_dict(checkpoint['warm_optimizer_state_dict'])
    joint_optimizer.load_state_dict(checkpoint['joint_optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from {CHECKPOINT} at epoch {init_epoch}")

else:
    init_epoch = 0


# Dataloaders
# Train
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=WORKERS)
train_push_loader = DataLoader(train_push_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)

# Validation
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)



# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
print(f'Size of training set: {len(train_loader.dataset)}')
print(f'Size of training push set: {len(train_push_loader.dataset)}')
print(f'Size of validation set: {len(val_loader.dataset)}')
print(f'Batch size: {BATCH_SIZE}')



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



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((NUM_TRAIN_EPOCHS, ))
val_losses = np.zeros_like(train_losses)

# Initialise metrics arrays
train_metrics = np.zeros((NUM_TRAIN_EPOCHS, 5))
val_metrics = np.zeros_like(train_metrics)

# Initialise best accuracy
best_accuracy = -np.inf
best_accuracy_push = -np.inf
best_accuracy_push_last = -np.inf



# Go through the number of Epochs
for epoch in range(init_epoch, NUM_TRAIN_EPOCHS):

    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Phase
    print("Training Phase")



    # Warming Phase
    # if epoch < num_warm_epochs:
    if epoch < NUM_WARM_EPOCHS:
        print("Training: Warm Phase")
        warm_only(model=ppnet_model, last_layer_fixed=LAST_LAYER_FIXED)
        metrics_dict = model_train(
            model=ppnet_model, 
            dataloader=train_loader,
            device=DEVICE,
            optimizer=warm_optimizer,
            class_specific=class_specific,
            coefs=COEFS,
            subtractive_margin=SUBTRACTIVE_MARGIN,
            use_ortho_loss=False
        )
    
    
    # Secondary Warming Phase
    # elif epoch >= num_warm_epochs and epoch - num_warm_epochs < num_secondary_warm_epochs:
    elif epoch >= NUM_WARM_EPOCHS and epoch - NUM_WARM_EPOCHS < NUM_SECONDARY_WARM_EPOCHS:
        print("Training: Secondary Warm Phase")
        warm_pre_offset(model=ppnet_model, last_layer_fixed=LAST_LAYER_FIXED)

        if DATASET == 'stanford_dogs':
            warm_lr_scheduler.step()
        
        metrics_dict = model_train(
            model=ppnet_model,
            dataloader=train_loader,
            device=DEVICE,
            optimizer=warm_pre_offset_optimizer,
            class_specific=class_specific,
            coefs=COEFS,
            subtractive_margin=SUBTRACTIVE_MARGIN,
            use_ortho_loss=False
        )


    # Joint Phase
    else:
        print("Training: Joint Phase")

        # if epoch == num_warm_epochs + num_secondary_warm_epochs:
        if epoch == NUM_WARM_EPOCHS + NUM_SECONDARY_WARM_EPOCHS:
            ppnet_model.module.initialize_offset_weights()

        joint(model=ppnet_model, last_layer_fixed=LAST_LAYER_FIXED)
        joint_lr_scheduler.step()

        metrics_dict = model_train(
            model=ppnet_model,
            dataloader=train_loader,
            device=DEVICE,
            optimizer=joint_optimizer,
            class_specific=class_specific,
            coefs=COEFS,
            subtractive_margin=SUBTRACTIVE_MARGIN,
            use_ortho_loss=True
        )


    # Print metrics
    print_metrics(metrics_dict=metrics_dict)

    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = metrics_dict["run_avg_loss"]
    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_tr_losses.npy")
    np.save(file=fname, arr=train_losses, allow_pickle=True)


    # Train Metrics
    # Accuracy
    train_metrics[epoch, 0] = metrics_dict['accuracy']
    # Recall
    train_metrics[epoch, 1] = metrics_dict['recall']
    # Precision
    train_metrics[epoch, 2] = metrics_dict['precision']
    # F1-Score
    train_metrics[epoch, 3] = metrics_dict['f1']
    # ROC AUC
    # train_metrics[epoch, 4] = metrics_dict['auc']

    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", metrics_dict["run_avg_loss"], global_step=epoch)
    tbwritter.add_scalar("acc/train", metrics_dict['accuracy'], global_step=epoch)
    tbwritter.add_scalar("rec/train", metrics_dict['recall'], global_step=epoch)
    tbwritter.add_scalar("prec/train", metrics_dict['precision'], global_step=epoch)
    tbwritter.add_scalar("f1/train", metrics_dict['f1'], global_step=epoch)
    # tbwritter.add_scalar("auc/train", metrics_dict['auc'], global_step=epoch)



    # Log to W&B
    wandb_tr_metrics = {
        "loss/train":metrics_dict["run_avg_loss"],
        "acc/train":metrics_dict['accuracy'],
        "rec/train":metrics_dict['recall'],
        "prec/train":metrics_dict['precision'],
        "f1/train":metrics_dict['f1'],
        "epoch/train":epoch
    }
    wandb.log(wandb_tr_metrics)



    # Validation Phase
    print("Validation Phase")
    metrics_dict = model_validation(
        model=ppnet_model,
        dataloader=val_loader,
        class_specific=class_specific,
        subtractive_margin=SUBTRACTIVE_MARGIN
    )

    # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu, target_accu=0.70, log=log)
    
    # Print metrics
    print_metrics(metrics_dict=metrics_dict)

    # Append values to the arrays
    # Validation Loss
    val_losses[epoch] = metrics_dict["run_avg_loss"]
    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_val_losses.npy")
    np.save(file=fname, arr=val_losses, allow_pickle=True)


    # Validation Metrics
    # Accuracy
    val_metrics[epoch, 0] = metrics_dict['accuracy']
    # Recall
    val_metrics[epoch, 1] = metrics_dict['recall']
    # Precision
    val_metrics[epoch, 2] = metrics_dict['precision']
    # F1-Score
    val_metrics[epoch, 3] = metrics_dict['f1']
    # ROC AUC
    # val_metrics[epoch, 4] = metrics_dict['auc']

    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_val_metrics.npy")
    np.save(file=fname, arr=val_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/val", metrics_dict["run_avg_loss"], global_step=epoch)
    tbwritter.add_scalar("acc/val", metrics_dict['accuracy'], global_step=epoch)
    tbwritter.add_scalar("rec/val", metrics_dict['recall'], global_step=epoch)
    tbwritter.add_scalar("prec/val", metrics_dict['precision'], global_step=epoch)
    tbwritter.add_scalar("f1/val", metrics_dict['f1'], global_step=epoch)
    # tbwritter.add_scalar("auc/val", metrics_dict['auc'], global_step=epoch)



    # Log to W&B
    wandb_val_metrics = {
        "loss/val":metrics_dict["run_avg_loss"],
        "acc/val":metrics_dict['accuracy'],
        "rec/val":metrics_dict['recall'],
        "prec/val":metrics_dict['precision'],
        "f1/val":metrics_dict['f1'],
        "epoch/val":epoch
    }
    wandb.log(wandb_val_metrics)



    # Log model's parameters and gradients to W&B
    wandb.watch(ppnet_model)



    # Save checkpoint
    if metrics_dict['accuracy'] > best_accuracy:

        print(f"Accuracy increased from {best_accuracy} to {metrics_dict['accuracy']}. Saving new model...")

        # Model path
        model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
        save_dict = {
            'epoch':epoch,
            'model_state_dict':ppnet_model.state_dict(),
            'warm_optimizer_state_dict':warm_optimizer.state_dict(),
            'joint_optimizer_state_dict':joint_optimizer.state_dict(),
            'run_avg_loss': metrics_dict["run_avg_loss"],
        }
        torch.save(save_dict, model_path)
        print(f"Successfully saved at: {model_path}")

        # Update best accuracy value
        best_accuracy = metrics_dict['accuracy']



    # PUSH START and PUSH EPOCHS
    # if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
    if (epoch == PUSH_START and PUSH_START < 20) or (epoch >= PUSH_START and epoch in PUSH_EPOCHS):
        print("Pushing Phase")
        push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_model, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=saved_prototypes_dir, # if not None, prototypes will be saved here
            epoch_number=None, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True
        )


        metrics_dict = model_validation(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)
        print_metrics(metrics_dict=metrics_dict)
        
        # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu, target_accu=0.70, log=log)
        # Save checkpoint
        if metrics_dict['accuracy'] > best_accuracy_push:

            print(f"Accuracy increased from {best_accuracy_push} to {metrics_dict['accuracy']}. Saving new model...")

            # Model path
            model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
            save_dict = {
                'epoch':epoch,
                'model_state_dict':ppnet_model.state_dict(),
                'warm_optimizer_state_dict':warm_optimizer.state_dict(),
                'joint_optimizer_state_dict':joint_optimizer.state_dict(),
                'run_avg_loss': metrics_dict["run_avg_loss"],
            }
            torch.save(save_dict, model_path)
            print(f"Successfully saved at: {model_path}")

            # Update best accuracy value
            best_accuracy_push = metrics_dict['accuracy']





        # If we intend to optimize last layer as well
        # if not last_layer_fixed:
        if not LAST_LAYER_FIXED:
            print("Optimizing last layer...")
            last_only(model=ppnet_model, last_layer_fixed=LAST_LAYER_FIXED)

            for i in range(20):
                print(f'Step {i+1} of {20}')

                print("Training")
                metrics_dict = model_train(
                    model=ppnet_model,
                    dataloader=train_loader,
                    device=DEVICE,
                    optimizer=last_layer_optimizer,
                    class_specific=class_specific,
                    coefs=COEFS,
                    subtractive_margin=SUBTRACTIVE_MARGIN
                )
                print_metrics(metrics_dict=metrics_dict)



                print("Validation")
                metrics_dict = model_validation(
                    model=ppnet_model,
                    dataloader=val_loader,
                    device=DEVICE,
                    class_specific=class_specific
                )
                print_metrics(metrics_dict=metrics_dict)

                # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu, target_accu=0.70, log=log)
                # Save checkpoint
                if metrics_dict['accuracy'] > best_accuracy_push_last:

                    print(f"Accuracy increased from {best_accuracy_push_last} to {metrics_dict['accuracy']}. Saving new model...")

                    # Model path
                    model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
                    save_dict = {
                        'epoch':epoch,
                        'model_state_dict':ppnet_model.state_dict(),
                        'warm_optimizer_state_dict':warm_optimizer.state_dict(),
                        'joint_optimizer_state_dict':joint_optimizer.state_dict(),
                        'run_avg_loss': metrics_dict["run_avg_loss"],
                    }
                    torch.save(save_dict, model_path)
                    print(f"Successfully saved at: {model_path}")

                    # Update best accuracy value
                    best_accuracy_push_last = metrics_dict['accuracy']



# Finish statement and W&B
wandb.finish()
print("Finished.")
