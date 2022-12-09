# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/train_and_test.py
# Imports
import numpy as np
from tqdm import tqdm

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# PyTorch Imports
import torch



# Function: List of Distances
def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)



# Function: Train or Test Phase
def run_model(model, dataloader, mode, device, optimizer=None, class_specific=True, use_l1_mask=True, coefs=None):

    # Assert mode
    assert mode in ("train", "validation", "test"), "Please provide a valid model mode (train, validation or test)."

    # Check if we are in training mode
    if mode == "train":
        assert optimizer is not None, "If the model is in training mode, you should provide an optimizer."


    # Some control variables
    # TODO: Erase uppon review
    # start = time.time()
    # n_examples = 0
    # n_correct = 0
    # n_batches = 0
    # Initialise lists to compute scores
    y_true = np.empty((0), int)
    y_pred = torch.empty(0, dtype=torch.int32, device=device)
    # Save scores after softmax for roc auc
    y_scores = torch.empty(0, dtype=torch.float, device=device)
    
        
    # Total Cross Entropy Loss & Total Cluster Cost
    total_cross_entropy = 0
    total_cluster_cost = 0


    # Separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0


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
            
            # We pass image by the model    
            logits, min_distances = model(images)
            # print(logits.type(), logits)
            # print(labels.type(), labels)

            # We first compute the CrossEntropy Loss
            cross_entropy = torch.nn.functional.cross_entropy(logits, labels.long())

            # Check this condition
            if class_specific:
                # max_dist = (model.module.prototype_shape[1] * model.module.prototype_shape[2] * model.module.prototype_shape[3])
                max_dist = (model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3])


                # Note: prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # Calculate cluster cost
                # TODO: Erase this part uppon review
                # prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                # prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).to(device)
                # labels_ = torch.reshape(labels, (-1,)).to(torch.int64)
                # print(labels_.shape)
                # print(labels_.dtype)
                # print(labels_)

                # Note: We had to convert labels to torch.int64 to allow slicing
                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,labels.to(torch.int64)]).to(device)
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # Calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # Calculate avg cluster cost
                avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                # Check this condition
                if use_l1_mask:
                    # l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    # l1_mask = 1 - torch.t(model.module.prototype_class_identity).to(device)
                    l1_mask = 1 - torch.t(model.prototype_class_identity).to(device)
                    # l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    # l1 = model.module.last_layer.weight.norm(p=1)
                    l1 = model.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                # l1 = model.module.last_layer.weight.norm(p=1)
                l1 = model.last_layer.weight.norm(p=1)


            # Compute performance metrics
            # TODO: Erase original code stuff uppon review
            # _, predicted = torch.max(logits.data, 1)
            # n_examples += labels.size(0)
            # n_correct += (predicted == label).sum().item()
            # n_batches += 1

            # Using Softmax
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            y_scores = torch.cat((y_scores, s_logits))
            s_logits = torch.argmax(s_logits, dim=1)
            y_pred = torch.cat((y_pred, s_logits))


            # Total Losses and Costs
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()



        # Compute final loss value for train or validation
        if mode in ("train", "validation"):
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1

            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1



            # Update Total Running Loss
            total_loss += loss.item()


            # Perform backpropagation (if training)
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        # Remove variables from memory
        del images
        del labels
        del logits
        del s_logits
        del min_distances


    # Some log prints
    # print('\ttime: \t{0}'.format(time.time() -  start))
    # print('Cross Entropy Loss: \t{0}'.format(total_cross_entropy / n_batches))
    # print('Cluster Loss: \t{0}'.format(total_cluster_cost / n_batches))
    ce_loss = total_cross_entropy / len(dataloader)
    # print('Cross Entropy Loss: \t{0}'.format(ce_loss))
    cluster_loss = total_cluster_cost / len(dataloader)
    # print('Cluster Loss: \t{0}'.format(cluster_loss))


    # Specific log prints for "class_specific" option
    if class_specific:
        # print('Separation Cost:\t{0}'.format(total_separation_cost / n_batches))
        # print('Average Separation Cost:\t{0}'.format(total_avg_separation_cost / n_batches))
        sep_cost = total_separation_cost / len(dataloader)
        # print('Separation Cost:\t{0}'.format(sep_cost))
        avg_sep_cost = total_avg_separation_cost / len(dataloader)
        # print('Average Separation Cost:\t{0}'.format(avg_sep_cost))
    else:
        sep_cost = 0.0
        avg_sep_cost = 0.0
    

    # Total Epoch Loss
    run_avg_loss = total_loss / len(dataloader)
    # print('Total Loss:\t{0}'.format(run_avg_loss))


    # Compute performance metrics
    # Get the necessary data
    y_pred = y_pred.cpu().detach().numpy()
    y_scores = y_scores.cpu().detach().numpy()

    # Accuracy, Recall, Precision, F1 and AUC
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    # auc = roc_auc_score(y_true=y_true, y_score=y_scores[:, 1], average='micro')

    # Print performance metrics
    # print('Accuracy: \t\t{0}%'.format(n_correct / n_examples * 100))
    # print('Accuracy: \t\t{0}%'.format(accuracy))
    # print('Recall: \t\t{0}%'.format(recall))
    # print('Precision: \t\t{0}%'.format(precision))
    # print('F1: \t\t{0}%'.format(f1))
    # print('AUC: \t\t{0}%'.format(auc))


    # Get L1
    # print('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    # print('L1: \t\t{0}'.format(model.last_layer.weight.norm(p=1).item()))
    l1 = model.last_layer.weight.norm(p=1).item()


    # Get prototypes
    # p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()


    # Compute prototype average pair distance
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    
    # print('Prototype Average Distance Pair: \t{0}'.format(p_avg_pair_dist.item()))
    p_avg_pair_dist_ = p_avg_pair_dist.item()


    # Create a metrics dictionary
    metrics_dict = {
        "accuracy":accuracy,
        "recall":recall,
        "precision":precision,
        "f1":f1,
        "run_avg_loss":run_avg_loss,
        "sep_cost":sep_cost,
        "avg_sep_cost":avg_sep_cost,
        "ce_loss":ce_loss,
        "cluster_loss":cluster_loss,
        "l1":l1,
        "p_avg_pair_dist":p_avg_pair_dist_
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

    # Separation Cost
    sep_cost = metrics_dict["sep_cost"]

    # Average Separation Cost
    avg_sep_cost = metrics_dict["avg_sep_cost"]

    # Cross Entropy Loss
    ce_loss = metrics_dict["ce_loss"]

    # Cluster Loss
    cluster_loss = metrics_dict["cluster_loss"]

    # L1 Distance
    l1 = metrics_dict["l1"]

    # Prototype Pair Average Distance
    p_avg_pair_dist = metrics_dict["p_avg_pair_dist"]


    # Print metrics
    print('Accuracy: {0}'.format(accuracy))
    print('Recall: {0}'.format(recall))
    print('Precision: {0}'.format(precision))
    print('F1: {0}'.format(f1))
    print('Total Loss: {0}'.format(run_avg_loss))
    print('Separation Cost: {0}'.format(sep_cost))
    print('Average Separation Cost: {0}'.format(avg_sep_cost))
    print('Cross Entropy Loss: {0}'.format(ce_loss))
    print('Cluster Loss: {0}'.format(cluster_loss))
    print('L1: {0}'.format(l1))
    print('Prototype Average Distance Pair: {0}'.format(p_avg_pair_dist))

    return



# Function: Train
def model_train(model, dataloader, device, optimizer, class_specific=False, coefs=None):

    # Put model in training mode
    model.train()

    return run_model(model=model, dataloader=dataloader, mode="train", device=device, optimizer=optimizer, class_specific=class_specific, coefs=coefs)



# Function: Validation
def model_validation(model, dataloader, device, class_specific=False):

    # Put model in evaluation mode
    model.eval()

    return run_model(model=model, dataloader=dataloader, mode="validation", device=device, optimizer=None, class_specific=class_specific)



# Function: Test
def model_test(model, dataloader, device, class_specific=False):

    # Put model in evaluation mode
    model.eval()

    return run_model(model=model, dataloader=dataloader, mode="test", device=device, optimizer=None, class_specific=class_specific)



# Function: Activate gradients on parameters for last-layer training phase
def last_only(model):
    # for p in model.module.features.parameters():
    for p in model.features.parameters():
        p.requires_grad = False

    # for p in model.module.add_on_layers.parameters():
    for p in model.add_on_layers.parameters():
        p.requires_grad = False

    # model.module.prototype_vectors.requires_grad = False
    model.prototype_vectors.requires_grad = False
    # for p in model.module.last_layer.parameters():
    for p in model.last_layer.parameters():
        p.requires_grad = True

    # print('\tlast layer')



# Function: Activate gradients on parameters for warm-training phase
def warm_only(model):
    # for p in model.module.features.parameters():
    for p in model.features.parameters():
        p.requires_grad = False
    
    # for p in model.module.add_on_layers.parameters():
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    
    # model.module.prototype_vectors.requires_grad = True
    model.prototype_vectors.requires_grad = True
    
    # for p in model.module.last_layer.parameters():
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    # print('\twarm')



# Function: Activate gradients on parameters for joint-training phase
def joint(model):
    # for p in model.module.features.parameters():
    for p in model.features.parameters():
        p.requires_grad = True

    # for p in model.module.add_on_layers.parameters():
    for p in model.add_on_layers.parameters():
        p.requires_grad = True

    # model.module.prototype_vectors.requires_grad = True
    model.prototype_vectors.requires_grad = True

    # for p in model.module.last_layer.parameters():
    for p in model.last_layer.parameters():
        p.requires_grad = True

    # print('\tjoint')



# Function: Get model predictions (i.e., inference)
def model_predict(model, in_data):


    # Get the logits and the minimum distances with the model in inference mode
    logits, min_distances = model(in_data)


    # Using Softmax: Apply softmax on logits to get the predicted scores
    s_logits = torch.nn.Softmax(dim=1)(logits)
    predicted_scores = s_logits.cpu().detach().numpy()

    # Get predicted label using argmax
    s_label = torch.argmax(s_logits, dim=1)
    predicted_label = s_label.cpu().detach().numpy()


    return predicted_label, predicted_scores
