# Imports
import os
import shutil
import numpy as np
import argparse
import re
import datetime

# PyTorch Imports
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
from data_utilities import preprocess_input_function
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

# m 
parser.add_argument('--m', nargs=1, type=float, default=None)

# Subtractive margin
# subtractive_margin = True
parser.add_argument('--subtractive_margin', action="store_true", default=True)

# Using deformable convolution
parser.add_argument('--using_deform', nargs=1, type=str, default=None)

# Top-K
parser.add_argument('--topk_k', nargs=1, type=int, default=None)

# Deformable Convolution Hidden Channels
parser.add_argument('--deformable_conv_hidden_channels', nargs=1, type=int, default=None)

# Number of Prototypes
parser.add_argument('--num_prototypes', nargs=1, type=int, default=None)

# Dilation
parser.add_argument('--dilation', nargs=1, type=float, default=2)

# Incorrect class connection
parser.add_argument('--incorrect_class_connection', nargs=1, type=float, default=0)

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
# push_start = 10
parser.add_argument('--push_start', type=int, default=10, help="Push start.")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Number of training epochs
# num_train_epochs = 1000
parser.add_argument('--num_train_epochs', type=int, default=300, help="Number of training epochs.")

# Number of warm epochs
# num_warm_epochs = 5
parser.add_argument('--num_warm_epochs', type=int, default=5, help="Number of warm epochs.")

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


# Settings - START
base_architecture = 'resnet50'
img_size = 224



# Full set: './datasets/CUB_200_2011/'
# Cropped set: './datasets/cub200_cropped/'
# Stanford dogs: './datasets/stanford_dogs/'
data_path = './datasets/CUB_200_2011/'
#120 classes in stanford_dogs, 200 in CUB_200_2011
if 'stanford_dogs' in data_path:
    num_classes = 120
else:
    num_classes = 200

train_dir = data_path + 'train/'
# Cropped set: train_cropped & test_cropped
# Full set: train & test
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75















num_train_epochs = 31
num_warm_epochs = 5
num_secondary_warm_epochs = 5
push_start = 20

push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

# Settings - END



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

# Save frquency
SAVE_FREQ = args.save_freq

# Resize (data transforms)
RESIZE_OPT = args.resize


# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(OUTPUT_DIR, DATASET.lower(), BASE_ARCHITECTURE.lower(), timestamp)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)





args = parser.parse_args()


m = args.m[0]
rand_seed = args.rand_seed[0]
last_layer_fixed = args.last_layer_fixed[0] == 'True'
subtractive_margin = args.subtractive_margin[0] == 'True'
using_deform = args.using_deform[0] == 'True'
topk_k = args.topk_k[0]
deformable_conv_hidden_channels = args.deformable_conv_hidden_channels[0]
num_prototypes = args.num_prototypes[0]

dilation = args.dilation
incorrect_class_connection = args.incorrect_class_connection[0]

print("---- USING DEFORMATION: ", using_deform)
print("Margin set to: ", m)
print("last_layer_fixed set to: {}".format(last_layer_fixed))
print("subtractive_margin set to: {}".format(subtractive_margin))
print("topk_k set to: {}".format(topk_k))
print("num_prototypes set to: {}".format(num_prototypes))
print("incorrect_class_connection: {}".format(incorrect_class_connection))
print("deformable_conv_hidden_channels: {}".format(deformable_conv_hidden_channels))



from settings import img_size, experiment_run, base_architecture

if num_prototypes is None:
    num_prototypes = 1200

if 'resnet34' in base_architecture:
    prototype_shape = (num_prototypes, 512, 2, 2)
    add_on_layers_type = 'upsample'
elif 'resnet152' in base_architecture:
    prototype_shape = (num_prototypes, 2048, 2, 2)
    add_on_layers_type = 'upsample'
elif 'resnet50' in base_architecture:
    prototype_shape = (num_prototypes, 2048, 2, 2)
    add_on_layers_type = 'upsample'
elif 'densenet121' in base_architecture:
    prototype_shape = (num_prototypes, 1024, 2, 2)
    add_on_layers_type = 'upsample'
elif 'densenet161' in base_architecture:
    prototype_shape = (num_prototypes, 2208, 2, 2)
    add_on_layers_type = 'upsample'
else:
    prototype_shape = (num_prototypes, 512, 2, 2)
    add_on_layers_type = 'upsample'
print("Add on layers type: ", add_on_layers_type)


base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

from settings import train_dir, test_dir, train_push_dir

# model_dir = './saved_models/' + base_architecture + '/' + train_dir + '/' + experiment_run + '/'
# makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
# makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

from settings import train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

if 'stanford_dogs' in train_dir:
    num_classes = 120
else:
    num_classes = 200
log("{} classes".format(num_classes))

if 'augmented' not in train_dir:
    print("Using online augmentation")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomAffine(degrees=(-25, 25), shear=15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
else:
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                            pretrained=True, img_size=img_size,
                            prototype_shape=prototype_shape,
                            num_classes=num_classes, topk_k=topk_k, m=m,
                            add_on_layers_type=add_on_layers_type,
                            using_deform=using_deform,
                            incorrect_class_connection=incorrect_class_connection,
                            deformable_conv_hidden_channels=deformable_conv_hidden_channels,
                            prototype_dilation=2)
    
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
    joint_optimizer_lrs['features'] = 1e-5
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.conv_offset.parameters(), 'lr': joint_optimizer_lrs['conv_offset']},
 {'params': ppnet.last_layer.parameters(), 'lr': joint_optimizer_lrs['joint_last_layer_lr']}
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.2)
log("joint_optimizer_lrs: ")
log(str(joint_optimizer_lrs))

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
log("warm_optimizer_lrs: ")
log(str(warm_optimizer_lrs))

from settings import warm_pre_offset_optimizer_lrs
if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
    warm_pre_offset_optimizer_lrs['features'] = 1e-5
warm_pre_offset_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_pre_offset_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_pre_offset_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.features.parameters(), 'lr': warm_pre_offset_optimizer_lrs['features'], 'weight_decay': 1e-3},
]
warm_pre_offset_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs)

warm_lr_scheduler = None
if 'stanford_dogs' in train_dir:
    warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)
    log("warm_pre_offset_optimizer_lrs: ")
    log(str(warm_pre_offset_optimizer_lrs))

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs
# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_warm_epochs, num_train_epochs, push_epochs, \
                    num_secondary_warm_epochs, push_start

# train the model
log('start training')

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=False)
    elif epoch >= num_warm_epochs and epoch - num_warm_epochs < num_secondary_warm_epochs:
        tnt.warm_pre_offset(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        if 'stanford_dogs' in train_dir:
            warm_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_pre_offset_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=False)
    else:
        if epoch == num_warm_epochs + num_secondary_warm_epochs:
            ppnet_multi.module.initialize_offset_weights()
        tnt.joint(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=True)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log, subtractive_margin=subtractive_margin)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)

    if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)

        if not last_layer_fixed:
            tnt.last_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, 
                            subtractive_margin=subtractive_margin)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
logclose()

