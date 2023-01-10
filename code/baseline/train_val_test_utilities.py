# Based on: https://github.com/cfchen-duke/ProtoPNet/blob/master/train_and_test.py
# Imports
import numpy as np
from tqdm import tqdm

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# PyTorch Imports
import torch



# Function: Train or Test Phase
def run_model(model, dataloader, mode, device, optimizer=None):

    # Assert mode
    assert mode in ("train", "validation", "test"), "Please provide a valid model mode (train, validation or test)."

    # Check if we are in training mode
    if mode == "train":
        assert optimizer is not None, "If the model is in training mode, you should provide an optimizer."


    # Some control variables
    y_true = np.empty((0), int)
    y_pred = torch.empty(0, dtype=torch.int32, device="cpu")

    # Total Cross Entropy Loss
    total_cross_entropy = 0

    # Total Loss
    total_loss = 0


    # Iterate through the dataloader
    for _, (images, labels) in enumerate(tqdm(dataloader)):

        # Concatenate lists
        y_true = np.append(y_true, labels.numpy(), axis=0)
        

        # Put data into device
        images, labels = images.to(device), labels.to(device)


        # Note: torch.enable_grad() has no effect outside of no_grad()
        if mode == "train":
            grad_req = torch.enable_grad()
        
        else:
            grad_req = torch.no_grad()


        # Get model's outputs
        with grad_req:
            
            # We pass the image(s) by the model    
            logits = model(images)

            # We first compute the CrossEntropy Loss
            cross_entropy_loss = torch.nn.functional.cross_entropy(logits, labels.long())


            # Using Softmax
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits).cpu().detach()
            s_logits = torch.argmax(s_logits, dim=1)
            y_pred = torch.cat((y_pred, s_logits))


            # Total Losses and Costs
            total_cross_entropy += cross_entropy_loss.item()



        # Compute final loss value for train or validation
        if mode in ("train", "validation"):

            # Update Total Running Loss
            total_loss += cross_entropy_loss.item()


            # Perform backpropagation (if training)
            if mode == "train":
                optimizer.zero_grad()
                cross_entropy_loss.backward()
                optimizer.step()


        # Remove variables from memory
        del images
        del labels
        del logits
        del s_logits



    # Loss information
    # Cross-entropy loss
    ce_loss = total_cross_entropy / len(dataloader)
    
    # Total epoch Loss
    run_avg_loss = total_loss / len(dataloader)


    # Compute performance metrics
    # Get the necessary data
    y_pred = y_pred.cpu().detach().numpy()

    # Accuracy, Recall, Precision, F1 and AUC
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')


    # Create a metrics dictionary
    metrics_dict = {
        "accuracy":accuracy,
        "recall":recall,
        "precision":precision,
        "f1":f1,
        "run_avg_loss":run_avg_loss,
        "ce_loss":ce_loss
    }

    return metrics_dict



# Function: Print Metrics
def print_metrics(metrics_dict):

    # Accuracy
    accuracy = metrics_dict["accuracy"]

    # Recall
    recall = metrics_dict["recall"]

    # Precision
    precision = metrics_dict["precision"]
    
    # F1-Score
    f1 = metrics_dict["f1"]

    # Loss
    run_avg_loss = metrics_dict["run_avg_loss"]

    # Cross Entropy Loss
    ce_loss = metrics_dict["ce_loss"]


    # Print metrics
    print('Accuracy: {0}'.format(accuracy))
    print('Recall: {0}'.format(recall))
    print('Precision: {0}'.format(precision))
    print('F1: {0}'.format(f1))
    print('Total Loss: {0}'.format(run_avg_loss))
    print('Cross Entropy Loss: {0}'.format(ce_loss))

    return



# Function: Train
def model_train(model, dataloader, device, optimizer):

    # Put model in training mode
    model.train()

    return run_model(model=model, dataloader=dataloader, mode="train", device=device, optimizer=optimizer)



# Function: Validation
def model_validation(model, dataloader, device):

    # Put model in evaluation mode
    model.eval()

    return run_model(model=model, dataloader=dataloader, mode="validation", device=device, optimizer=None)



# Function: Test
def model_test(model, dataloader, device):

    # Put model in evaluation mode
    model.eval()

    return run_model(model=model, dataloader=dataloader, mode="test", device=device, optimizer=None)



# Function: Activate gradients on parameters for last-layer training phase
def last_only(model):
    for p in model.features.parameters():
        p.requires_grad = False

    for p in model.last_layer.parameters():
        p.requires_grad = True



# Function: Activate gradients on parameters for warm-training phase
def warm_only(model):
    for p in model.features.parameters():
        p.requires_grad = True
    
    for p in model.last_layer.parameters():
        p.requires_grad = False



# Function: Activate gradients on parameters for joint-training phase
def joint(model):
    for p in model.features.parameters():
        p.requires_grad = True

    for p in model.last_layer.parameters():
        p.requires_grad = True



# Function: Get model predictions (i.e., inference)
def model_predict(model, in_data):


    # Get the logits with the model in inference mode
    logits = model(in_data)

    # Using Softmax: Apply softmax on logits to get the predicted scores
    s_logits = torch.nn.Softmax(dim=1)(logits)
    predicted_scores = s_logits.cpu().detach().numpy()

    # Get predicted label using argmax
    s_label = torch.argmax(s_logits, dim=1)
    predicted_label = s_label.cpu().detach().numpy()


    return predicted_label, predicted_scores
