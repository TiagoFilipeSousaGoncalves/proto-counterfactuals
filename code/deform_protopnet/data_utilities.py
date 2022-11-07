# Imports
import torch



# Function: Apply normalisation to a given input function
def preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    assert x.size(1) == 3

    y = torch.zeros_like(x)

    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]

    return y



# Function: Allocate new tensor like x and apply the normalization used in the pretrained model
def preprocess_input_function(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    return preprocess(x=x, mean=mean, std=std)



# Function: Denormalise input(s), i.e., in this case images
def undo_preprocess(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    
    assert x.size(1) == 3
    
    y = torch.zeros_like(x)
    
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    
    return y



# Function: Allocate new tensor like x and undo the normalization used in the pretrained model
def undo_preprocess_input_function(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    
    return undo_preprocess(x=x, mean=mean, std=std)
