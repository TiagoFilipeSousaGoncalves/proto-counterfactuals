# Imports
from PIL import Image

# PyTorch Imports
import torch
from torch.autograd import Variable
import torch.utils.data



# Function: Generate image counterfactual
def get_image_counterfactual(image_path, ppnet_model, device, transforms):

    # Load the image and labels
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transforms(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.to(device)

    # Run inference with ppnet
    logits, _ = ppnet_model(images_test)
    s_logits = torch.nn.Softmax(dim=1)(logits)
    sorted_indices = torch.argsort(s_logits, dim=1)
    # idx_max = torch.argmax(s_logits, dim=1)
    # print(sorted_indices[0])

    # Get prediction and counterfactual
    label_pred = sorted_indices[0][-1].item()
    # print(label_pred, idx_max[0].item())
    counterfactual_pred = sorted_indices[0][-2].item()
    # print(counterfactual_pred)
    


    return label_pred, counterfactual_pred



# Function: Generate image features
def generate_image_features(image_path, ppnet_model, device, transforms):


    # Load the image and labels
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transforms(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    image_test = img_variable.to(device)

    # Run inference with ppnet
    conv_output, _ = ppnet_model.push_forward(image_test)


    return conv_output
