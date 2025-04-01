# Imports
import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm

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
from model_utilities import construct_PPNet
from train_val_test_utilities import model_predict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")
    parser.add_argument('--dataset', type=str, required=True, choices=["cub2002011", "papila", "ph2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")
    parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, resnet18, vgg19.')
    parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms.")
    parser.add_argument('--prototype_activation_function', type=str, default='log', help="Prototype activation function.")
    parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")
    args = parser.parse_args()



    # Constants
    DATA_DIR = args.data_dir
    DATASET = args.dataset
    BASE_ARCHITECTURE = args.base_architecture
    RESULTS_DIR = args.results_dir
    WORKERS = args.num_workers
    PROTOTYPE_ACTIVATION_FUNCTION = args.prototype_activation_function
    IMG_SIZE = args.img_size
    ADD_ON_LAYERS_TYPE = args.add_on_layers_type


    # Inference directory
    inference_dir = os.path.join(RESULTS_DIR, "analysis", "counterfac-inf")
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)


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
    # train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=WORKERS)
    # val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=WORKERS)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, pin_memory=False, num_workers=WORKERS)



    # Results and Weights
    weights_dir = os.path.join(RESULTS_DIR, "weights")


    # Choose GPU
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {DEVICE}")



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


    # Load model weights
    # model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
    # model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
    model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
    model_weights = torch.load(model_path_push_last, map_location=DEVICE)
    ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
    # print(f"Model weights loaded with success from {model_path_push_last}.")


    # Put model into evaluation mode
    ppnet_model.eval()


    # Perform inference and iterate through the dataloader
    inference_dict = {
        "Image Index":list(),
        "Ground-Truth Label":list(),
        "Predicted Label":list(),
        "Predicted Scores":list(),
        "Counterfactuals":list()
    }


    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader)):

            # Put data into GPU or CPU
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Get scores and prediction
            predicted_label, predicted_scores = model_predict(model=ppnet_model, in_data=images)
            ground_truth_label = labels.cpu().detach().numpy()

            # Get the counterfactual (i.e., the second most activated class)
            # deb_gt = np.argsort(predicted_scores[0].copy())[-1]
            cntr_fac = np.argsort(predicted_scores[0].copy())[-2]


            # Add information to the dictionary
            inference_dict["Image Index"].append(idx)
            # print(f"Image Index: {idx}")
            inference_dict["Ground-Truth Label"].append(ground_truth_label[0])
            # print(f"Ground-Truth Label: {ground_truth_label[0]}")
            inference_dict["Predicted Label"].append(predicted_label[0])
            # print(f"Predicted Label: {predicted_label[0]} (debug: {deb_gt})")
            inference_dict["Predicted Scores"].append(predicted_scores[0])
            # print(f"Predicted Scores: {predicted_scores[0]}")
            inference_dict["Counterfactuals"].append(cntr_fac)
            # print(f"Counterfactuals: {cntr_fac}")

            # Debug print
            # print(f"Evolution of the inference dictionary:\n {inference_dict}\n")



    # Check if old analysis.csv file exists
    pkl_path = os.path.join(inference_dir, "inference_results.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)

    # Convert this into .PKL
    with open(pkl_path, 'wb') as f:
        pickle.dump(inference_dict, f, -1)