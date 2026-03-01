import argparse
import datetime
import os
import random

import numpy as np
import torch
import torchvision
import wandb
from data_utilities import (CUB2002011Dataset, PAPILADataset, PH2Dataset,
                            preprocess_input_function)
from model_utilities import construct_PPNet
from prototypes_utilities import push_prototypes
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchinfo import summary
from train_val_test_utilities import (joint, last_only, model_train,
                                      model_validation, print_metrics,
                                      warm_only)


# Function: Set random seed
def set_seed(seed=42):

    # Random
    random.seed(seed)

    # Python Environment
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")
    parser.add_argument('--dataset', type=str, required=True, choices=["cub2002011", "papila", "ph2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")
    parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, resnet18, vgg19.')
    parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")
    parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")
    parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")
    parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")
    parser.add_argument('--joint_optimizer_lrs', type=dict, default={'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}, help="Joint optimizer learning rates.")
    parser.add_argument('--joint_lr_step_size', type=int, default=5, help="Joint learning rate step size.")
    parser.add_argument('--warm_optimizer_lrs', type=dict, default={'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}, help="Warm optimizer learning rates.")
    parser.add_argument('--last_layer_optimizer_lr', type=float, default=1e-4, help="Last layer optimizer learning rate.")
    parser.add_argument('--coefs', type=dict, default={'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4,}, help="Loss coeficients.")
    parser.add_argument('--push_start', type=int, default=10, help="Push start.")
    parser.add_argument('--resize', type=str, choices=["direct_resize", "resizeshortest_randomcrop"], default="direct_resize", help="Resize data transformation")
    parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")
    parser.add_argument('--num_train_epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--num_warm_epochs', type=int, default=5, help="Number of warm epochs.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency (in number of epochs) to save the model")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from which to resume training")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="Fold for cross-validation (if needed).")
    parser.add_argument('--timestamp', type=str, required=False, help="Timestamp for results saving (if needed).")
    parser.add_argument('--seed', type=int, default=42, help="Set the random seed for determinism.")
    args = parser.parse_args()


    # Set seed
    set_seed(seed=args.seed)

    # Log in to W&B Account
    wandb.login()


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

    # Initialize timestamp for results saving
    timestamp = args.timestamp if args.timestamp else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for fold in args.folds:
        results_dir = os.path.join(OUTPUT_DIR, DATASET.lower(), "baseline", BASE_ARCHITECTURE.lower(), timestamp, f"fold_{fold}")
        os.makedirs(results_dir, exist_ok=True)

        # Save training parameters
        with open(os.path.join(results_dir, "train_params.txt"), "w") as f:
            f.write(str(args))


        # Set the W&B project
        wandb.init(
            project="proto-counterfactuals",
            name=timestamp,
            config={
                "architecture": "ppnet-" + BASE_ARCHITECTURE.lower(),
                "dataset": DATASET.lower(),
                "train-epochs": NUM_TRAIN_EPOCHS,
                "fold":fold
            }
        )


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
            torchvision.transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=45),
            torchvision.transforms.ElasticTransform(),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomPerspective(),
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
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=MEAN, std=STD)
        ])


        if DATASET == "cub2002011":
            train_set = CUB2002011Dataset(
                data_path=DATA_DIR,
                fold=fold,
                split="train",
                transform=train_transforms
            )
            train_push_set = CUB2002011Dataset(
                data_path=DATA_DIR,
                fold=fold,
                split="train",
                transform=train_push_transforms
            )
            val_set = CUB2002011Dataset(
                data_path=DATA_DIR,
                fold=fold,
                split='val',
                transform=val_transforms
            )
            NUM_CLASSES = len(train_set.labels_dict)


        elif DATASET == "papila":
            train_set = PAPILADataset(
                data_path=DATA_DIR,
                fold=fold,
                split="train",
                transform=train_transforms
            )
            train_push_set = PAPILADataset(
                data_path=DATA_DIR,
                fold=fold,
                split="train",
                transform=train_push_transforms
            )
            val_set = PAPILADataset(
                data_path=DATA_DIR,
                fold=fold,
                split="val",
                transform=val_transforms
            )
            NUM_CLASSES = len(np.unique(train_set.images_labels))


        elif DATASET == "ph2":
            train_set = PH2Dataset(
                data_path=DATA_DIR,
                fold=fold,
                split="train",
                transform=train_transforms
            )
            train_push_set = PH2Dataset(
                data_path=DATA_DIR,
                fold=fold,
                split="train",
                transform=train_push_transforms
            )
            val_set = PH2Dataset(
                data_path=DATA_DIR,
                fold=fold,
                split="val",
                transform=val_transforms
            )
            NUM_CLASSES = len(train_set.diagnosis_dict)



        # Results and Weights
        weights_dir = os.path.join(results_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)


        # History Files
        history_dir = os.path.join(results_dir, "history")
        os.makedirs(history_dir, exist_ok=True)



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


        # Construct the Model
        ppnet_model = construct_PPNet(
            base_architecture=BASE_ARCHITECTURE,
            pretrained=True,
            img_size=IMG_SIZE,
            prototype_shape=PROTOTYPE_SHAPE,
            num_classes=NUM_CLASSES,
            prototype_activation_function=PROTOTYPE_ACTIVATION_FUNCTION,
            add_on_layers_type=ADD_ON_LAYERS_TYPE
        )

        # if prototype_activation_function == 'linear':
        #     ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)

        # Define if the model is class specific or not
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


        # Log model's parameters and gradients to W&B
        wandb.watch(ppnet_model)

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
        val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)



        # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
        print(f'Size of training set: {len(train_loader.dataset)}')
        print(f'Size of training push set: {len(train_push_loader.dataset)}')
        print(f'Size of validation set: {len(val_loader.dataset)}')
        print(f'Batch size: {BATCH_SIZE}')



        # Define some extra paths and directories
        # Directory for saved prototypes
        saved_prototypes_dir = os.path.join(weights_dir, 'prototypes')
        os.makedirs(saved_prototypes_dir, exist_ok=True)


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


            if epoch < NUM_WARM_EPOCHS:
                print("Training: Warm Phase")
                warm_only(model=ppnet_model)
                metrics_dict = model_train(model=ppnet_model, dataloader=train_loader, device=DEVICE, optimizer=warm_optimizer, class_specific=class_specific, coefs=COEFS)


            else:
                print("Training: Joint Phase")
                joint(model=ppnet_model)
                joint_lr_scheduler.step()
                metrics_dict = model_train(model=ppnet_model, dataloader=train_loader, device=DEVICE, optimizer=joint_optimizer, class_specific=class_specific, coefs=COEFS)


            # Print metrics
            print_metrics(metrics_dict=metrics_dict)

            # Append values to the arrays
            # Train Loss
            train_losses[epoch] = metrics_dict["run_avg_loss"]
            # Save it to directory
            fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_tr_losses.npy")
            np.save(file=fname, arr=train_losses, allow_pickle=True)


            # Train Metrics
            # Acc
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


            # Log to W&B
            wandb_tr_metrics = {
                "loss_train":metrics_dict["run_avg_loss"],
                "acc_train":metrics_dict['accuracy'],
                "rec_train":metrics_dict['recall'],
                "prec_train":metrics_dict['precision'],
                "f1_train":metrics_dict['f1'],
                "epoch_train":epoch
            }
            wandb.log(wandb_tr_metrics)



            # Validation Phase
            print("Validation Phase")
            metrics_dict = model_validation(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)

            # Print metrics
            print_metrics(metrics_dict=metrics_dict)

            # Append values to the arrays
            # Validation Loss
            val_losses[epoch] = metrics_dict["run_avg_loss"]
            # Save it to directory
            fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_val_losses.npy")
            np.save(file=fname, arr=val_losses, allow_pickle=True)


            # Train Metrics
            # Acc
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



            # Log to W&B
            wandb_val_metrics = {
                "loss_val":metrics_dict["run_avg_loss"],
                "acc_val":metrics_dict['accuracy'],
                "rec_val":metrics_dict['recall'],
                "prec_val":metrics_dict['precision'],
                "f1_val":metrics_dict['f1'],
                "epoch_val":epoch
            }
            wandb.log(wandb_val_metrics)


            # save.save_model_w_condition(model=ppnet_model, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu, target_accu=0.70, log=log)
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
            if epoch >= PUSH_START and epoch in PUSH_EPOCHS:
                print("Pushing Phase")
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

                metrics_dict = model_validation(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)


                # Print metrics
                print_metrics(metrics_dict=metrics_dict)


                # save.save_model_w_condition(model=ppnet_model, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu, target_accu=0.70)
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



                # If the protoype activation function is not linear
                if PROTOTYPE_ACTIVATION_FUNCTION != 'linear':
                    print("Optimizing last layer...")
                    last_only(model=ppnet_model)

                    for i in range(20):
                        print(f'Step {i+1} of {20}')
                        print("Training")
                        metrics_dict = model_train(model=ppnet_model, dataloader=train_loader, device=DEVICE, optimizer=last_layer_optimizer, class_specific=class_specific, coefs=COEFS)
                        print_metrics(metrics_dict=metrics_dict)

                        print("Validation")
                        metrics_dict = model_validation(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)
                        print_metrics(metrics_dict=metrics_dict)

                        # Save checkpoint
                        # save.save_model_w_condition(model=ppnet_model, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu, target_accu=0.70)
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
