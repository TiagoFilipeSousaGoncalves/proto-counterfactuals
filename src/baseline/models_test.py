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

# Project Imports
from data_utilities import CUB2002011Dataset, PAPILADataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import DenseNet, ResNet, VGG
from train_val_test_utilities import model_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the data set.")
    parser.add_argument('--dataset', type=str, required=True, choices=["cub2002011", "papila", "ph2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")
    parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19".')
    parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")
    parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")
    args = parser.parse_args()


    DATA_DIR = args.data_dir
    DATASET = args.dataset
    BASE_ARCHITECTURE = args.base_architecture
    RESULTS_DIR = args.results_dir
    WORKERS = args.num_workers
    BATCH_SIZE = args.batchsize
    IMG_SIZE = args.img_size

    # Mean and STD to Normalize the inputs into pretrained models
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]



    # Test Transforms
    eval_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])



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
            split="val",
            augmented=False,
            transform=eval_transforms
        )

        test_set = CUB2002011Dataset(
            data_path=DATA_DIR,
            split="test",
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
            transform=eval_transforms
        )

        # Validation Dataset
        val_set = STANFORDCARSDataset(
            data_path=DATA_DIR,
            cars_subset="cars_test",
            augmented=False,
            cropped=True,
            transform=eval_transforms
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


    # Construct the Model
    if BASE_ARCHITECTURE.lower() in ("densenet121", "densenet161"):
        baseline_model = DenseNet(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)
    elif BASE_ARCHITECTURE.lower() in ("resnet34", "resnet152"):
        baseline_model = ResNet(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)
    else:
        baseline_model = VGG(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)


    # Put model into DEVICE (CPU or GPU)
    baseline_model = baseline_model.to(DEVICE)


    # Load model weights (model_path_push does not exist in the baseline architectures)
    model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
    # model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
    model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")



    # Create a report to append these results
    if os.path.exists(os.path.join(RESULTS_DIR, "acc_report.txt")):
        os.remove(os.path.join(RESULTS_DIR, "acc_report.txt"))

    report = open(os.path.join(RESULTS_DIR, "acc_report.txt"), "at")


    # Iterate through these model weights types
    for model_fname in [model_path, model_path_push_last]:
        model_weights = torch.load(model_fname, map_location=DEVICE)
        baseline_model.load_state_dict(model_weights['model_state_dict'], strict=True)
        report.write(f"Model weights loaded with success from {model_fname}.\n")

        # Get performance metrics
        train_metrics_dict = model_test(model=baseline_model, dataloader=train_loader, device=DEVICE)
        train_acc = train_metrics_dict["accuracy"]
        report.write(f"Train Accuracy: {train_acc}\n")

        val_metrics_dict = model_test(model=baseline_model, dataloader=val_loader, device=DEVICE)
        val_acc = train_metrics_dict["accuracy"]
        report.write(f"Validation Accuracy: {val_acc}\n")

        test_metrics_dict = model_test(model=baseline_model, dataloader=test_loader, device=DEVICE)
        test_accuracy = test_metrics_dict["accuracy"]
        report.write(f"Test Accuracy: {test_accuracy}\n")