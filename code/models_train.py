# Imports
import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from data_utilities import preprocess_input_function, CUB2002011Dataset
from model_utilities import construct_PPNet
from prototypes_utilities import push_prototypes
from train_and_test_utilities import train, test, last_only, warm_only, joint



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011"], help="Data set: CUB2002011")

# Model
# base_architecture = 'vgg19'
parser.add_argument('--base_architecture', type=str, required=True, choices=["vgg19"], help='Base architecture: vgg19, ')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")

# Prototype shape
# prototype_shape = (2000, 128, 1, 1)
parser.add_argument('--prototype_shape', type=tuple, default=(2000, 128, 1, 1), help="Prototype shape.")

# Number of classes
# num_classes = 200
parser.add_argument('--num_classes', type=int, default=200, help="Number of classes.")

# Prototype Activation Function
# prototype_activation_function = 'log'
parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")

# Add on layers type
# add_on_layers_type = 'regular'
parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")

# Joint optimizer learning rates
# joint_optimizer_lrs = {'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}
parser.add_argument('--joint_optimizer_lrs', type=dict, default={'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}, help="Joint optimizer learning rates.")

# Joint learning rate step size
# joint_lr_step_size = 5
parser.add_argument('--joint_lr_step_size', type=int, default=5, help="Joint learning rate step size.")

# Warm optimizer learning rates
# warm_optimizer_lrs = {'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}
parser.add_argument('--warm_optimizer_lrs', type=dict, default={'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}, help="Warm optimizer learning rates.")

# Last layer optimizer learning rate
# last_layer_optimizer_lr = 1e-4
parser.add_argument('--last_layer_optimizer_lr', type=float, default=1e-4, help="Last layer optimizer learning rate.")

# Loss coeficients
# coefs = {'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}
parser.add_argument('--coefs', type=dict, default={'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}, help="Loss coeficients.")

# Push start
# push_start = 10
parser.add_argument('--push_start', type=int, default=10, help="Push start.")

# Resize
parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Number of training epochs
# num_train_epochs = 1000
parser.add_argument('--num_train_epochs', type=int, default=1000, help="Number of training epochs.")

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



# CLI ProtoPNet (Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/settings.py)
# experiment_run = '003'
# data_path = './datasets/cub200_cropped/'
# train_dir = data_path + 'train_cropped_augmented/'
# test_dir = data_path + 'test_cropped/'
# train_push_dir = data_path + 'train_cropped/'
# train_batch_size = 80
# test_batch_size = 100
# train_push_batch_size = 75



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

# Prototype shape
PROTOTYPE_SHAPE = args.prototype_shape

# Number of classes
NUM_CLASSES = args.num_classes

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


# Save training parameters
with open(os.path.join(results_dir, "train_params.txt"), "w") as f:
    f.write(str(args))



# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE



# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if RESIZE_OPT == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if RESIZE_OPT == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Push Transforms
train_push_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if RESIZE_OPT == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Validation Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMG_SIZE if RESIZE_OPT == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop(IMG_SIZE if RESIZE_OPT == 'resizeshortest_randomcrop' else (IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


# Dataset
if DATASET == "CUB2002011":
    # Train Dataset
    train_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub_200_2011", "processed_data", "train", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub_200_2011", "source_data", "classes.txt"),
        transform=train_transforms
    )

    # Train Push Dataset (Prototypes)
    train_push_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub_200_2011", "processed_data", "train", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub_200_2011", "source_data", "classes.txt"),
         transform=train_push_transforms
    )

    # Validation Dataset
    val_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub_200_2011", "processed_data", "test", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub_200_2011", "source_data", "classes.txt"),
        transform=val_transforms
    )





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



class_specific = True



# Define optimizers and learning rate schedulers
# Joint Optimizer Specs
joint_optimizer_specs = [

    {'params': ppnet_model.features.parameters(), 'lr': JOINT_OPTIMIZER_LRS['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet_model.add_on_layers.parameters(), 'lr': JOINT_OPTIMIZER_LRS['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet_model.prototype_vectors, 'lr': JOINT_OPTIMIZER_LRS['prototype_vectors']},

]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=JOINT_LR_STEP_SIZE, gamma=0.1)


# Warm Optimizer Learning Rates
warm_optimizer_specs = [

    {'params': ppnet_model.add_on_layers.parameters(), 'lr': WARM_OPTIMIZER_LRS['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet_model.prototype_vectors, 'lr': WARM_OPTIMIZER_LRS['prototype_vectors']},

]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)


# Last Layer Optimizer
last_layer_optimizer_specs = [

    {'params': ppnet_model.last_layer.parameters(), 'lr': LAST_LAYER_OPTIMIZER_LR}
    
]
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



# Class weights for loss
if args.classweights:
    classes = np.array(range(NUM_CLASSES))
    cw = compute_class_weight('balanced', classes=classes, y=np.array(train_set.images_labels))
    cw = torch.from_numpy(cw).float().to(DEVICE)
    print(f"Using class weights {cw}")
else:
    cw = None




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
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=WORKERS)



# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
print(f'Size of training set: {len(train_loader.dataset)}')
print(f'Size of training push set: {len(train_push_loader.dataset)}')
print(f'Size of validation set:: {len(val_loader.dataset)}')
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



# Go through the number of Epochs
for epoch in tqdm(range(init_epoch, NUM_TRAIN_EPOCHS)):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Phase
    print("Training Phase")
    
    # Initialise lists to compute scores
    y_train_true = np.empty((0), int)
    y_train_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)
    y_train_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc


    # Running train loss
    run_train_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)


    # Put model in training mode
    ppnet_model.train()


    if epoch < NUM_WARM_EPOCHS:
        warm_only(model=ppnet_model)
        _ = train(model=ppnet_model, dataloader=train_loader, device=DEVICE, optimizer=warm_optimizer, class_specific=class_specific, coefs=COEFS)


    else:
        joint(model=ppnet_model)
        joint_lr_scheduler.step()
        _ = train(model=ppnet_model, dataloader=train_loader, device=DEVICE, optimizer=joint_optimizer, class_specific=class_specific, coefs=COEFS)



    # Validation Phase 
    print("Validation Phase")
    accu = test(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)
    # save.save_model_w_condition(model=ppnet_model, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu, target_accu=0.70, log=log)
    # Save checkpoint
    # Model path
    model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
    save_dict = {
        'epoch':epoch,
        'model_state_dict':ppnet_model.state_dict(),
        'warm_optimizer_state_dict':warm_optimizer.state_dict(),
        'joint_optimizer_state_dict':joint_optimizer.state_dict(),
        # 'loss': avg_train_loss,
    }
    torch.save(save_dict, model_path)
    print(f"Successfully saved at: {model_path}")


    # PUSH START and PUSH EPOCHS
    if epoch >= PUSH_START and epoch in PUSH_EPOCHS:
        push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_model, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=saved_prototypes_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True
            )
        
        accu = test(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)
        # save.save_model_w_condition(model=ppnet_model, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu, target_accu=0.70)


        # If the protoype activation function is not linear
        if PROTOTYPE_ACTIVATION_FUNCTION != 'linear':
            last_only(model=ppnet_model)
            
            for i in range(20):
                print('iteration: \t{0}'.format(i))
                _ = train(model=ppnet_model, dataloader=train_loader, optimizer=last_layer_optimizer, class_specific=class_specific, coefs=COEFS)
                
                accu = test(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)
                # save.save_model_w_condition(model=ppnet_model, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu, target_accu=0.70)
   










"""
    # Iterate through dataloader
    for images, labels in tqdm(train_loader):
        # Concatenate lists
        y_train_true = np.append(y_train_true, labels.numpy(), axis=0)

        # Move data and model to GPU (or not)
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad(set_to_none=True)


        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)

        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)

        # Update batch losses
        run_train_loss += loss

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()

        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        y_train_scores = torch.cat((y_train_scores, s_logits))
        s_logits = torch.argmax(s_logits, dim=1)
        y_train_pred = torch.cat((y_train_pred, s_logits))


    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)
    

    # Compute Train Metrics
    y_train_pred = y_train_pred.cpu().detach().numpy()
    y_train_scores = y_train_scores.cpu().detach().numpy()
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_auc = roc_auc_score(y_true=y_train_true, y_score=y_train_scores[:, 1], average='micro')

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")


    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = avg_train_loss
    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_losses.npy")
    np.save(file=fname, arr=train_losses, allow_pickle=True)


    # Train Metrics
    # Acc
    train_metrics[epoch, 0] = train_acc
    # Recall
    train_metrics[epoch, 1] = train_recall
    # Precision
    train_metrics[epoch, 2] = train_precision
    # F1-Score
    train_metrics[epoch, 3] = train_f1
    # ROC AUC
    train_metrics[epoch, 4] = train_auc

    # Save it to directory
    fname = os.path.join(history_dir, f"{model_name}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", avg_train_loss, global_step=epoch)
    tbwritter.add_scalar("acc/train", train_acc, global_step=epoch)
    tbwritter.add_scalar("rec/train", train_recall, global_step=epoch)
    tbwritter.add_scalar("prec/train", train_precision, global_step=epoch)
    tbwritter.add_scalar("f1/train", train_f1, global_step=epoch)
    tbwritter.add_scalar("auc/train", train_auc, global_step=epoch)

    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss


    # Validation Loop
    print("Validation Phase")


    # Initialise lists to compute scores
    y_val_true = np.empty((0), int)
    y_val_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)
    y_val_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc

    # Running train loss
    run_val_loss = 0.0

    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels in tqdm(val_loader):
            y_val_true = np.append(y_val_true, labels.numpy(), axis=0)

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = VAL_LOSS(logits, labels)
            
            # Update batch losses
            run_val_loss += loss


            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)                        
            y_val_scores = torch.cat((y_val_scores, s_logits))
            s_logits = torch.argmax(s_logits, dim=1)
            y_val_pred = torch.cat((y_val_pred, s_logits))

        

        # Compute Average Validation Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Validation Accuracy
        y_val_pred = y_val_pred.cpu().detach().numpy()
        y_val_scores = y_val_scores.cpu().detach().numpy()
        val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
        val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_auc = roc_auc_score(y_true=y_val_true, y_score=y_val_scores[:, 1], average='micro')

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")

        # Append values to the arrays
        # Validation Loss
        val_losses[epoch] = avg_val_loss
        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_losses.npy")
        np.save(file=fname, arr=val_losses, allow_pickle=True)


        # Train Metrics
        # Acc
        val_metrics[epoch, 0] = val_acc
        # Recall
        val_metrics[epoch, 1] = val_recall
        # Precision
        val_metrics[epoch, 2] = val_precision
        # F1-Score
        val_metrics[epoch, 3] = val_f1
        # ROC AUC
        val_metrics[epoch, 4] = val_auc

        # Save it to directory
        fname = os.path.join(history_dir, f"{model_name}_val_metrics.npy")
        np.save(file=fname, arr=val_metrics, allow_pickle=True)

        # Plot to Tensorboard
        tbwritter.add_scalar("loss/val", avg_val_loss, global_step=epoch)
        tbwritter.add_scalar("acc/val", val_acc, global_step=epoch)
        tbwritter.add_scalar("rec/val", val_recall, global_step=epoch)
        tbwritter.add_scalar("prec/val", val_precision, global_step=epoch)
        tbwritter.add_scalar("f1/val", val_f1, global_step=epoch)
        tbwritter.add_scalar("auc/val", val_auc, global_step=epoch)

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{model_name}_{dataset.lower()}_best.pt")
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMISER.state_dict(),
                'loss': avg_train_loss,
            }
            torch.save(save_dict, model_path)

            print(f"Successfully saved at: {model_path}")


        # Checkpoint loop/condition
        if epoch % save_freq == 0 and epoch > 0:

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{model_name}_{dataset.lower()}_{epoch:04}.pt")

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMISER.state_dict(),
                'loss': avg_train_loss,
            }
            torch.save(save_dict, model_path)

"""

# Finish statement
print("Finished.")
