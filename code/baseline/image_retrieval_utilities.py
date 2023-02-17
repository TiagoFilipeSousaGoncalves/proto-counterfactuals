# Imports
from PIL import Image

# PyTorch Imports
import torch
from torch.autograd import Variable
import torch.utils.data



# Function: Generate image counterfactual
def get_image_counterfactual(image_path, baseline_model, device, transforms):

    # Put model into evaluation mode
    baseline_model.eval()

    # Load the image and labels
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transforms(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.to(device)

    # Run inference with baseline model
    logits = baseline_model(images_test)
    s_logits = torch.nn.Softmax(dim=1)(logits)
    sorted_indices = torch.argsort(s_logits, dim=1)

    # Get prediction and counterfactual
    label_pred = sorted_indices[0][-1].item()
    counterfactual_pred = sorted_indices[0][-2].item()


    return label_pred, counterfactual_pred



# Function: Generate image features
def generate_image_features(image_path, baseline_model, device, transforms, feature_space):

    assert feature_space in ("conv_features"), "Please provide a valid feature space ('conv_features')."


    # Put model into evaluation mode
    baseline_model.eval()


    # Load the image and labels
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transforms(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    image_test = img_variable.to(device)


    # Run inference with ppnet
    features = baseline_model.conv_features(image_test)


    return features



# Function: Generate image prediction
def get_image_prediction(image_path, baseline_model, device, transforms):

    # Put model into evaluation mode
    baseline_model.eval()

    # Load the image and labels
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transforms(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.to(device)

    # Run inference with ppnet
    logits = baseline_model(images_test)
    s_logits = torch.nn.Softmax(dim=1)(logits)
    sorted_indices = torch.argsort(s_logits, dim=1)
    

    # Get prediction
    label_pred = sorted_indices[0][-1].item()


    return label_pred
