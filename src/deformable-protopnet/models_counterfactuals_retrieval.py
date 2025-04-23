# Note: Provide an input image, get the nearest counterfactual and compare prototypes



# Imports
import os
import argparse
import pickle
import numpy as np

# PyTorch Imports
import torch
import torchvision

# Project Imports
from data_utilities import CUB2002011Dataset, PAPILADataset, PH2Dataset, STANFORDCARSDataset
from image_retrieval_utilities import generate_image_features, get_image_counterfactual, get_image_prediction
from model_utilities import construct_PPNet



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")
    parser.add_argument('--dataset', type=str, required=True, choices=["cub2002011", "papila", "ph2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")
    parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet50", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, densenet161, resnet34, resnet50, resnet152, vgg16, vgg19.')
    parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation.")
    parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms.")
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--subtractive_margin', default=True, action="store_true")
    parser.add_argument('--using_deform', default=True, action="store_true")
    parser.add_argument('--topk_k', type=int, default=1)
    parser.add_argument('--deformable_conv_hidden_channels', type=int, default=128)
    parser.add_argument('--num_prototypes', type=int, default=1200)
    parser.add_argument('--dilation', type=float, default=2)
    parser.add_argument('--incorrect_class_connection', type=float, default=-0.5)
    parser.add_argument('--add_on_layers_type', type=str, default='regular', help="Add on layers type.")
    parser.add_argument('--last_layer_fixed', default=True, action="store_true")
    parser.add_argument("--classweights", default=False, action="store_true", help="Weight loss with class imbalance.")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU.")
    parser.add_argument("--generate_img_features", default=False, action="store_true", help="Generate features for the retrieval.")
    parser.add_argument("--feature_space", type=str, required=True, choices=["conv_features", "proto_features"], help="Feature space: convolutional features (conv_features) or prototype layer features (proto_features).")
    args = parser.parse_args()


    # Constants
    RESULTS_DIR = args.results_dir
    DATA_DIR = args.data_dir
    DATASET = args.dataset
    BASE_ARCHITECTURE = args.base_architecture
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
    GENERATE_FEATURES = args.generate_img_features
    FEATURE_SPACE = args.feature_space



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
        train_images_fpaths = train_set.images_names
        train_labels_dict = train_set.images_labels

        val_set = PH2Dataset(
            data_path=DATA_DIR,
            subset="val",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )
        val_images_fpaths = val_set.images_names
        val_labels_dict = val_set.images_labels

        test_set = PH2Dataset(
            data_path=DATA_DIR,
            subset="test",
            cropped=True,
            augmented=False,
            transform=eval_transforms
        )
        test_images_fpaths = test_set.images_names
        test_labels_dict = test_set.images_labels

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
    # print("Add on layers type: ", ADD_ON_LAYERS_TYPE)


    # Weights
    weights_dir = os.path.join(RESULTS_DIR, "weights")


    # Features 
    features_dir = os.path.join(RESULTS_DIR, "features", FEATURE_SPACE)
    if not os.path.isdir(features_dir):
        os.makedirs(features_dir)


    # Choose GPU
    DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {DEVICE}")


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

    # Load model weights (we load the weights that correspond to the last stage of training)
    # model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
    model_path_push = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push.pt")
    # model_path_push_last = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
    model_weights = torch.load(model_path_push, map_location=DEVICE)
    ppnet_model.load_state_dict(model_weights['model_state_dict'], strict=True)
    # print(f"Model weights loaded with success from: {model_path_push}.")


    # Create a local analysis path
    save_analysis_path = os.path.join(RESULTS_DIR, "analysis", "image-retrieval", FEATURE_SPACE)
    if not os.path.isdir(save_analysis_path):
        os.makedirs(save_analysis_path)


    # Get prototype shape
    prototype_shape = ppnet_model.prototype_shape



    # Generate images features (we will need these for the retrieval)
    # Note: We generate features for the entire database
    if GENERATE_FEATURES:
        for images_fpaths, labels_dict in zip([train_images_fpaths, val_images_fpaths, test_images_fpaths], [train_labels_dict, val_labels_dict, test_labels_dict]):
            for idx, eval_image_path in enumerate(images_fpaths):

                # Get image label
                if DATASET == 'cub2002011':
                    eval_image_folder = eval_image_path.split("/")[-2]
                    eval_image_label = labels_dict[eval_image_folder]
                elif DATASET in ('papila', 'ph2'):
                    eval_image_folder = eval_image_path.split("/")[-2]
                    eval_image_label = labels_dict[idx]


                    # Generate features
                    features = generate_image_features(
                        eval_image_path=eval_image_path,
                        ppnet_model=ppnet_model,
                        device=DEVICE,
                        eval_transforms=eval_transforms,
                        feature_space=FEATURE_SPACE
                    )


                    # Convert feature vector to NumPy
                    features = features.detach().cpu().numpy()

                    # Save this into disk
                    image_name = eval_image_path.split("/")[-1]
                    features_fname = image_name.split('.')[0] + '.npy'
                    features_fpath = os.path.join(features_dir, features_fname)
                    np.save(
                        file=features_fpath,
                        arr=features,
                        allow_pickle=True,
                        fix_imports=True
                    )



    # Analysis .CSV
    analysis_dict = {
        "Image":list(),
        "Image Label":list(),
        "Nearest Counterfactual":list(),
        "Nearest Counterfactual Label":list(),
    }



    # Query Image: Here, we will only use the images of the test set as query images
    # Go through all test image directories
    for images_fpaths, labels_dict in zip([test_images_fpaths], [test_labels_dict]):
        for idx, eval_image_path in enumerate(images_fpaths):
            if DATASET == 'cub2002011':
                eval_image_folder = eval_image_path.split("/")[-2]
                eval_image_label = labels_dict[eval_image_folder]
            elif DATASET in ('papila', 'ph2'):
                eval_image_folder = eval_image_path.split("/")[-2]
                eval_image_label = labels_dict[idx]


            # Get image counterfactual
            label_pred, counterfactual_pred = get_image_counterfactual(
                eval_image_path=eval_image_path,
                ppnet_model=ppnet_model,
                device=DEVICE,
                eval_transforms=eval_transforms
            )


            # Check if the predicted label is equal to the ground-truth label
            if int(eval_image_label) == int(label_pred):

                # Then we check for counterfactuals
                # First, we get the features, by loading the feature vectors
                if GENERATE_FEATURES:
                    image_name = eval_image_path.split("/")[-1]
                    eval_img_fts = np.load(os.path.join(features_dir, image_name.split('.')[0] + '.npy'), allow_pickle=True, fix_imports=True)
                else:
                    eval_img_fts = generate_image_features(
                        eval_image_path=eval_image_path,
                        ppnet_model=ppnet_model,
                        device=DEVICE,
                        eval_transforms=eval_transforms,
                        feature_space=FEATURE_SPACE
                    )
                    eval_img_fts = eval_img_fts.detach().cpu().numpy()


                # Create lists to append temporary values
                counter_imgs_fnames = list()
                distances = list()


                # Iterate again through the TRAIN images of the database
                for ctf_images_fpaths, ctf_labels_dict in zip([train_images_fpaths, val_images_fpaths], [train_labels_dict, val_labels_dict]):
                    for ctf_idx, ctf_image_path in enumerate(ctf_images_fpaths):
                        if DATASET == 'cub2002011':
                            ctf_image_folder = ctf_image_path.split("/")[-2]
                            ctf_image_label = ctf_labels_dict[ctf_image_folder]
                        elif DATASET in ('papila', 'ph2'):
                            ctf_image_folder = ctf_image_path.split("/")[-2]
                            ctf_image_label = ctf_labels_dict[ctf_idx]

                        

                        # We only evaluate in such cases
                        if int(ctf_image_label) == int(counterfactual_pred):

                            # Get the prediction of the model on this counterfactual
                            ctf_prediction = get_image_prediction(
                                eval_image_path=ctf_image_path,
                                ppnet_model=ppnet_model,
                                device=DEVICE,
                                eval_transforms=eval_transforms
                            )

                            # Only compute the distances to cases where both the ground-truth and the predicted label(s) of the counterfactual match
                            if int(ctf_prediction) == int(ctf_image_label):
                                
                                # Load the features of the counterfactual
                                if GENERATE_FEATURES:
                                    ctf_fname = ctf_image_path.split("/")[-1]
                                    ctf_fts = np.load(os.path.join(features_dir, ctf_fname.split('.')[0] + '.npy'), allow_pickle=True, fix_imports=True)
                                # Or generate the vector, anyway    
                                else:
                                    ctf_fts = generate_image_features(
                                        eval_image_path=ctf_image_path,
                                        ppnet_model=ppnet_model,
                                        device=DEVICE,
                                        eval_transforms=eval_transforms,
                                        feature_space=FEATURE_SPACE
                                    )
                                    ctf_fts = ctf_fts.detach().cpu().numpy()

                                    

                                # Compute the Euclidean Distance (L2-norm) between these feature vectors
                                distance_img_ctf = np.linalg.norm(eval_img_fts-ctf_fts)


                                # Append these to lists
                                counter_imgs_fnames.append(ctf_image_path)
                                distances.append(distance_img_ctf)
        


                # Add information to our data dictionary
                analysis_dict["Image"].append(eval_image_path)
                analysis_dict["Image Label"].append(int(eval_image_label))

                # We must be sure that we found at least one valid counterfactual
                if len(distances) > 0:
                    analysis_dict["Nearest Counterfactual"].append(counter_imgs_fnames[np.argmin(distances)])
                else:
                    analysis_dict["Nearest Counterfactual"].append("N/A")
                
                analysis_dict["Nearest Counterfactual Label"].append(int(counterfactual_pred))


    # Check if old analysis.csv file exists
    pkl_path = os.path.join(save_analysis_path, "analysis.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)

    # Convert this into .PKL
    with open(pkl_path, 'wb') as f:
        pickle.dump(analysis_dict, f, -1)