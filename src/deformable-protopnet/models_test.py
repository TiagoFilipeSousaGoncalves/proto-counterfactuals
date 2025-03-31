# Imports
import os
import argparse
import numpy as np

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from data_utilities import CUB2002011Dataset, PAPILADataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
from train_val_test_utilities import model_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")
    parser.add_argument('--dataset', type=str, required=True, choices=["cub2002011", "papila", "ph2", "STANFORDCARS"], help="Data set: cub2002011, papila, ph2, STANFORDCARS.")
    parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet50", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, densenet161, resnet34, resnet50, resnet152, vgg16, vgg19.')
    parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")
    parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--subtractive_margin', action="store_true")
    parser.add_argument('--using_deform', action="store_true")
    parser.add_argument('--topk_k', type=int, default=1)
    parser.add_argument('--deformable_conv_hidden_channels', type=int, default=128)
    parser.add_argument('--num_prototypes', type=int, default=1200)
    parser.add_argument('--dilation', type=float, default=2)
    parser.add_argument('--incorrect_class_connection', type=float, default=-0.5)
    parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")
    parser.add_argument('--last_layer_fixed', action="store_true")
    parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")
    args = parser.parse_args()

    # Constants
    DATA_DIR = args.data_dir
    DATASET = args.dataset
    BASE_ARCHITECTURE = args.base_architecture
    RESULTS_DIR = args.results_dir
    WORKERS = args.num_workers
    BATCH_SIZE = args.batchsize
    IMG_SIZE = args.img_size
    ADD_ON_LAYERS_TYPE = args.add_on_layers_type
    LAST_LAYER_FIXED = args.last_layer_fixed
    MARGIN = args.margin
    SUBTRACTIVE_MARGIN = args.subtractive_margin
    USING_DEFORM = args.using_deform
    TOPK_K = args.topk_k
    DEFORMABLE_CONV_HIDDEN_CHANNELS = args.deformable_conv_hidden_channels
    NUM_PROTOTYPES = args.num_prototypes
    DILATION = args.dilation
    INCORRECT_CLASS_CONNECTION = args.incorrect_class_connection

    # Load data
    # Mean and STD to Normalize the inputs into pretrained models
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]



    # Test Transforms
    eval_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])



    # Dataset
    # CUB2002011
    if DATASET == "cub2002011":
        train_set = CUB2002011Dataset(
            data_path=DATA_DIR,
            split="train",
            augmented=False,
            transform=eval_transforms
        )

        val_set = CUB2002011Dataset(
            data_path=DATA_DIR,
            split='val',
            augmented=False,
            transform=eval_transforms
        )

        test_set = CUB2002011Dataset(
            data_path=DATA_DIR,
            split='test',
            augmented=False,
            transform=eval_transforms
        )

        # Number of classes
        NUM_CLASSES = len(train_set.labels_dict)



    # PAPILA
    elif DATASET == "papila":
        train_set = PAPILADataset(
            data_path=DATA_DIR,
            subset="train",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )

        val_set = PAPILADataset(
            data_path=DATA_DIR,
            subset="val",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )

        test_set = PAPILADataset(
            data_path=DATA_DIR,
            subset="test",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )

        # Number of Classes
        NUM_CLASSES = len(np.unique(train_set.images_labels))



    # PH2
    elif DATASET == "ph2":
        train_set = PH2Dataset(
            data_path=DATA_DIR,
            subset="train",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )

        val_set = PH2Dataset(
            data_path=DATA_DIR,
            subset="val",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )

        test_set = PH2Dataset(
            data_path=DATA_DIR,
            subset="test",
            cropped=True,
            augmented=False,
            transform=eval_transforms
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



    # DataLoaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)



    # Results and Weights
    weights_dir = os.path.join(RESULTS_DIR, "weights")


    # Choose GPU
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")



    # Define the number of prototypes per class
    # if NUM_PROTOTYPES == -1:
    # NUM_PROTOTYPES_CLASS = NUM_PROTOTYPES
    NUM_PROTOTYPES_CLASS = int(NUM_CLASSES * 10)

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
        device=DEVICE,
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



    # Put model into DEVICE (CPU or GPU)
    ppnet_model = ppnet_model.to(DEVICE)


    # Load model weights
    model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
    model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")

    if not LAST_LAYER_FIXED:
        model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
    else:
        model_path_push_last = None


    # Create a report to append these results
    if os.path.exists(os.path.join(RESULTS_DIR, "acc_report.txt")):
        os.remove(os.path.join(RESULTS_DIR, "acc_report.txt"))

    report = open(os.path.join(RESULTS_DIR, "acc_report.txt"), "at")


    # Iterate through these model weights types
    for model_fname in [model_path, model_path_push, model_path_push_last]:

        if model_fname is not None:
            model_weights = torch.load(model_fname, map_location=DEVICE)
            ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
            report.write(f"Model weights loaded with success from {model_fname}.\n")


            # Get performance metrics
            train_metrics_dict = model_test(model=ppnet_model, dataloader=train_loader, device=DEVICE, class_specific=class_specific)
            train_acc = train_metrics_dict["accuracy"]
            report.write(f"Train Accuracy: {train_acc}\n")

            val_metrics_dict = model_test(model=ppnet_model, dataloader=val_loader, device=DEVICE, class_specific=class_specific)
            val_acc = val_metrics_dict["accuracy"]
            report.write(f"Validation Accuracy: {val_acc}\n")

            test_metrics_dict = model_test(model=ppnet_model, dataloader=test_loader, device=DEVICE, class_specific=class_specific)
            test_acc = test_metrics_dict["accuracy"]
            report.write(f"Test Accuracy: {test_acc}\n")