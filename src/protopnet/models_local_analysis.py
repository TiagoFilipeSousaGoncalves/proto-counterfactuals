# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/local_analysis.py
# Note: Finding the nearest prototypes to a test image

# Imports
import os
import argparse
import numpy as np
import pickle

# PyTorch Imports
import torch
import torchvision
import torch.utils.data

# Project Imports
from data_utilities import CUB2002011Dataset, PAPILADataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import construct_PPNet
from prototypes_retrieval_utilities import retrieve_image_prototypes



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


    # Load data
    # Mean and STD to Normalize the inputs into pretrained models
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]



    # Transforms
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
        train_images_fpaths = [os.path.join(train_set.data_path, "processed", train_set.split, "cropped", img_path) for img_path in train_set.images_fpaths]
        train_labels_dict = train_set.labels_dict

        val_set = CUB2002011Dataset(
            data_path=DATA_DIR,
            split='val',
            augmented=False,
            transform=eval_transforms
        )
        val_images_fpaths = [os.path.join(val_set.data_path, "processed", val_set.split, "cropped", img_path) for img_path in val_set.images_fpaths]
        val_labels_dict = val_set.labels_dict

        test_set = CUB2002011Dataset(
            data_path=DATA_DIR,
            split='test',
            augmented=False,
            transform=eval_transforms
        )
        test_images_fpaths = [os.path.join(test_set.data_path, "processed", test_set.split, "cropped", img_path) for img_path in test_set.images_fpaths]
        test_labels_dict = test_set.labels_dict

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
        train_images_fpaths = train_set.images_names
        train_labels_dict = train_set.images_labels

        val_set = PAPILADataset(
            data_path=DATA_DIR,
            subset="val",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )
        val_images_fpaths = val_set.images_names
        val_labels_dict = val_set.images_labels

        test_set = PAPILADataset(
            data_path=DATA_DIR,
            subset="test",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )
        test_images_fpaths = test_set.images_names
        test_labels_dict = test_set.images_labels

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


    # Weights
    weights_dir = os.path.join(RESULTS_DIR, "weights")

    # Prototypes
    # load_img_dir = os.path.join(weights_dir, 'prototypes')
    prototypes_img_dir = os.path.join(weights_dir, 'prototypes')


    # Choose GPU
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {DEVICE}")


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

    # Load model weights (we should read the last optimised layer)
    # model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
    # model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
    model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
    model_weights = torch.load(model_path_push_last, map_location=DEVICE)
    ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
    # print(f"Model weights loaded with success from: {model_path_push_last}.")


    # Create a local analysis path
    save_analysis_path = os.path.join(RESULTS_DIR, "analysis", "local")
    if not os.path.isdir(save_analysis_path):
        os.makedirs(save_analysis_path)


    # Get prototype shape
    prototype_shape = ppnet_model.prototype_shape

    # Get max distance
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]



    # Analysis .CSV
    analysis_dict = {
        "Image Filename":list(),
        "Ground-Truth Label":list(),
        "Predicted Label":list(),
        "Number of Prototypes Connected to the Class Identity":list(),
        "Top-10 Prototypes Class Identities":list()
    }


    # Go through all image directories
    for images_fpaths, labels_dict in zip([train_images_fpaths, val_images_fpaths, test_images_fpaths], [train_labels_dict, val_labels_dict, test_labels_dict]):
        for eval_image_path in images_fpaths:
            print(eval_image_path)

            # Get image label
            if DATASET == 'cub2002011':
                eval_image_folder = eval_image_path.split("/")[-2]
                eval_image_label = labels_dict[eval_image_folder]
            elif DATASET == 'papila':
                exit()

            # Create image analysis path
            image_analysis_path = os.path.join(save_analysis_path, eval_image_folder, eval_image_path.split('/')[-1])
            if not os.path.isdir(image_analysis_path):
                os.makedirs(image_analysis_path)

            # Analyse this image
            img_fname, gt_label, pred_label, nr_prototypes_cls_ident, topk_proto_cls_ident = retrieve_image_prototypes(
                image_analysis_path=image_analysis_path,
                prototypes_img_dir=prototypes_img_dir,
                ppnet_model=ppnet_model,
                device=DEVICE,
                eval_transforms=eval_transforms,
                eval_image_path=eval_image_path,
                eval_image_label=eval_image_label,
                norm_params={"mean":MEAN, "std":STD},
                img_size=IMG_SIZE
            )


            # Add information to our data dictionary
            analysis_dict["Image Filename"].append(img_fname)
            analysis_dict["Ground-Truth Label"].append(gt_label)
            analysis_dict["Predicted Label"].append(pred_label)
            analysis_dict["Number of Prototypes Connected to the Class Identity"].append(nr_prototypes_cls_ident)
            analysis_dict["Top-10 Prototypes Class Identities"].append(topk_proto_cls_ident)


    # Save data dictionary into a .PKL (check if old analysis.csv file exists)
    pkl_path = os.path.join(save_analysis_path, "analysis.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(analysis_dict, f, -1)