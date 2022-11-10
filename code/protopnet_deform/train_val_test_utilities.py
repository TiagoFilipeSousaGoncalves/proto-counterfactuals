# Imports
import numpy as np
from tqdm import tqdm

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# PyTorch Imports
import torch



# Function: Return model in a predefined phase (train, validation or test)
def run_model(model, dataloader, mode, device, optimizer=None, class_specific=True, use_l1_mask=True, coefs=None, subtractive_margin=True, use_ortho_loss=False):

    # Assert mode
    assert mode in ("train", "validation", "test"), "Please provide a valid model mode (train, validation or test)."

    # Check if we are in training mode
    if mode == "train":
        assert optimizer is not None, "If the model is in training mode, you should provide an optimizer."


    # TODO: Erase uppon review
    # is_train = optimizer is not None
    # n_examples = 0
    # n_correct = 0
    # n_batches = 0

    # Initialise lists to compute scores
    y_true = np.empty((0), int)
    y_pred = torch.empty(0, dtype=torch.int32, device=device)
    
    # Save scores after softmax for roc auc
    y_scores = torch.empty(0, dtype=torch.float, device=device)


    # Initialise variables to save loss and cost
    total_cross_entropy = 0
    total_cluster_cost = 0


    # Note: Separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_l2 = 0
    total_ortho_loss = 0
    max_offset = 0


    # If we use L1-Mask
    if use_l1_mask:
        l1_mask = 1 - torch.t(model.prototype_class_identity).to(device)
        l1 = (model.last_layer.weight * l1_mask).norm(p=1)
    else:
        l1 = model.last_layer.weight.norm(p=1)
    

    # Total Loss
    total_loss = 0


    # Iterate through dataloader
    for _, (images, labels) in enumerate(tqdm(dataloader)):

        # Concatenate lists
        y_true = np.append(y_true, labels.numpy(), axis=0)

        # Move data into GPU or CPU
        images, labels = images.to(device), labels.to(device)


        # Activate gradient computation
        # Note: torch.enable_grad() has no effect outside of no_grad()
        if mode == "train":
            grad_req = torch.enable_grad()
        else:
            grad_req = torch.no_grad()
        

        with grad_req:

            # Note: nn.Module has implemented __call__() function so no need to call .forward
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,labels]).to(device)
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class


            # If we are using subtractive margin
            if subtractive_margin:
                if mode == "train":
                    logits, additional_returns = model(images, is_train=True, prototypes_of_wrong_class=prototypes_of_wrong_class)
                else:
                    logits, additional_returns = model(images, is_train=False, prototypes_of_wrong_class=prototypes_of_wrong_class)
            else:
                if mode == "train":
                    logits, additional_returns = model(images, is_train=True, prototypes_of_wrong_class=None)
                else:
                    logits, additional_returns = model(images, is_train=False, prototypes_of_wrong_class=None)


            # Get additional returns in a proper way
            max_activations = additional_returns[0]
            marginless_logits = additional_returns[1]
            conv_features = additional_returns[2]


            with torch.no_grad():

                # Prototype shape
                prototype_shape = model.prototype_shape

                # Epsilon value
                epsilon_val = model.epsilon_val

                # Number of epsilon channels
                n_eps_channels = model.module.n_eps_channels

                # Pass convolutional features into x
                x = conv_features
                epsilon_channel_x = torch.ones(x.shape[0], n_eps_channels, x.shape[2], x.shape[3]) * epsilon_val
                epsilon_channel_x = epsilon_channel_x.to(device)
                x = torch.cat((x, epsilon_channel_x), -3)

                # Input vector length
                input_vector_length = model.input_vector_length
                normalizing_factor = (prototype_shape[-2] * prototype_shape[-1]) ** 0.5
                input_length = torch.sqrt(torch.sum(torch.square(x), dim=-3))
                input_length = input_length.view(input_length.size()[0], 1, input_length.size()[1], input_length.size()[2]) 
                input_normalized = input_vector_length * x / input_length
                input_normalized = input_normalized / normalizing_factor
                offsets = model.conv_offset(input_normalized)


            # Get several values again
            epsilon_val = model.epsilon_val
            n_eps_channels = model.n_eps_channels
            epsilon_channel_x = torch.ones(conv_features.shape[0], n_eps_channels, conv_features.shape[2], conv_features.shape[3]) * epsilon_val
            epsilon_channel_x = epsilon_channel_x.to(device)
            conv_features = torch.cat((conv_features, epsilon_channel_x), -3)

            # Compute CE loss
            cross_entropy = torch.nn.functional.cross_entropy(logits, labels)

            if class_specific:
                # Calculate cluster cost
                correct_class_prototype_activations, _ = torch.max(max_activations * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(correct_class_prototype_activations)

                # Calculate separation cost
                incorrect_class_prototype_activations, _ = torch.max(max_activations * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(incorrect_class_prototype_activations)

                # Calculate average cluster cost
                avg_separation_cost = torch.sum(max_activations * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                offset_l2 = offsets.norm()

            else:
                max_activations, _ = torch.max(max_activations, dim=1)
                cluster_cost = torch.mean(max_activations)
                l1 = model.last_layer.weight.norm(p=1)


            # TODO: Evaluation statistics
            # _, predicted = torch.max(marginless_logits.data, 1)
            # n_examples += target.size(0)
            # n_correct += (predicted == target).sum().item()
            # n_batches += 1

            # Using Softmax
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(marginless_logits.data)
            y_scores = torch.cat((y_scores, s_logits))
            s_logits = torch.argmax(s_logits, dim=1)
            y_pred = torch.cat((y_pred, s_logits))


            # Update losses and costs
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_l2 += offset_l2
            total_avg_separation_cost += avg_separation_cost.item()
            batch_max = torch.max(torch.abs(offsets))
            max_offset = torch.max(torch.Tensor([batch_max, max_offset]))


            # Compute keypoint-wise orthogonality loss, i.e. encourage each piece of a prototype to be orthogonal to the others
            orthogonalities = model.get_prototype_orthogonalities()
            orthogonality_loss = torch.norm(orthogonalities)
            total_ortho_loss += orthogonality_loss.item()


        # Compute gradient and do SGD step
        if mode in ("train", "validation"):
            if class_specific:
                if coefs is not None:
                    total_ortho_loss += orthogonality_loss.item()
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          + coefs['offset_bias_l2'] * offset_l2)
                    if use_ortho_loss:
                        loss += coefs['orthogonality_loss'] * orthogonality_loss
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


            if mode == "train":
                # Perform backpropagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()


        # FIXME: Check if we need these lines
        # del batch_max, predicted, max_activations
        # del offsets, conv_features, prototypes_of_correct_class, prototypes_of_wrong_class
        # del epsilon_channel_x, input_normalized, input_vector_length, x, additional_returns
        # del marginless_logits, offset_l2, cross_entropy, cluster_cost, separation_cost
        # del orthogonalities, orthogonality_loss, correct_class_prototype_activations
        # del incorrect_class_prototype_activations, avg_separation_cost, input_length


    # TODO: Erase uppon review
    # if use_ortho_loss:
    #     log('\tUsing ortho loss')


    # Create a metrics dictionary
    metrics_dict = dict()


    # Total Epoch Loss
    run_avg_loss = total_loss / len(dataloader)
    metrics_dict["run_avg_loss"] = run_avg_loss


    # Cross-entropy loss
    # log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    ce_loss = total_cross_entropy / len(dataloader)
    metrics_dict["ce_loss"] = ce_loss

    # Cluster loss
    # log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    cluster_loss = total_cluster_cost / len(dataloader)
    metrics_dict["cluster_loss"] = cluster_loss


    # If we have class specific
    if class_specific:
        # Separation cost
        # log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        sep_cost = total_separation_cost / len(dataloader)
        metrics_dict["sep_cost"] = sep_cost

        # Avg separation cost
        # log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
        avg_sep_cost = total_avg_separation_cost / len(dataloader)
        metrics_dict["avg_sep_cost"] = avg_sep_cost


    # Accuracy, Recall, Precision, F1 and AUC
    # Compute performance metrics
    # Get the necessary data
    y_pred = y_pred.cpu().detach().numpy()
    y_scores = y_scores.cpu().detach().numpy()

    # log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    # accuracy = n_correct / n_examples * 100
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics_dict["accuracy"] = accuracy
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    metrics_dict["recall"] = recall
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    metrics_dict["precision"] = precision
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    metrics_dict["f1"] = f1

    # Orthogonality loss
    # log('\torthogonality loss:\t{0}'.format(total_ortho_loss / n_batches))
    orthog_loss = total_ortho_loss / len(dataloader)
    metrics_dict["orthog_loss"] = orthog_loss

    # L1
    # log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    l1 = model.last_layer.weight.norm(p=1).item()
    metrics_dict["l1"] = l1
    
    # Avg L2
    # log('\tavg l2: \t\t{0}'.format(total_l2 / n_batches))
    avg_l2 = total_l2 / len(dataloader)
    metrics_dict["avg_l2"] = avg_l2


    # If we have coefficients
    if coefs is not None:

        # Avg L2 w/ weight
        # log('\tavg l2 with weight: \t\t{0}'.format(coefs['offset_bias_l2'] * total_l2 / n_batches))
        avg_l2_weight = coefs['offset_bias_l2'] * total_l2 / len(dataloader)
        metrics_dict["avg_l2_weight"] = avg_l2_weight

        # Orthogonality loss w/ weight
        # log('\torthogonality loss with weight:\t{0}'.format(coefs['orthogonality_loss'] * total_ortho_loss / n_batches))
        orthog_loss_weight = coefs['orthogonality_loss'] * total_ortho_loss / len(dataloader)
        metrics_dict["orthog_loss_weight"] = orthog_loss_weight


    # Max offset
    # log('\tmax offset: \t{0}'.format(max_offset))

    return metrics_dict



# Function: Print Metrics
def print_metrics(metrics_dict, class_specific, coefs):

    # Total loss
    run_avg_loss = metrics_dict["run_avg_loss"]
    print(f"Total loss: {run_avg_loss}")

    # Cross-entropy loss
    ce_loss = metrics_dict["ce_loss"]
    print('Cross-entropy loss: {0}'.format(ce_loss))

    # Cluster loss
    cluster_loss = metrics_dict["cluster_loss"]
    print('Cluster loss: {0}'.format(cluster_loss))


    # Separation cost and average separation cost
    if class_specific:
        sep_cost = metrics_dict["sep_cost"]
        avg_sep_cost = metrics_dict["avg_sep_cost"]
        print('Separation cost: {0}'.format(sep_cost))
        print('Average separation cost: {0}'.format(avg_sep_cost))


    # Accuracy, Recall, Precision, F1 and AUC
    accuracy = metrics_dict["accuracy"]
    recall = metrics_dict["recall"]
    precision = metrics_dict["precision"]
    f1 = metrics_dict["f1"]
    print('Accuracy: {0}'.format(accuracy))
    print('Recall: {0}'.format(recall))
    print('Precision: {0}'.format(precision))
    print('F1-Score: {0}'.format(f1))

    # Orthogonality loss
    orthog_loss = metrics_dict["orthog_loss"]
    print('Orthogonality loss:\t{0}'.format(orthog_loss))

    # L1
    l1 = metrics_dict["l1"] = l1
    print('L1: {0}'.format(l1))
    
    # Avg L2
    avg_l2 = metrics_dict["avg_l2"]
    print('Average L2: {0}'.format(avg_l2))


    # Average L2 w/ weight and Orthogonality loss w/ weight
    if coefs is not None:
        avg_l2_weight = metrics_dict["avg_l2_weight"]
        orthog_loss_weight = metrics_dict["orthog_loss_weight"]
        print('Average L2 with weight: {0}'.format(avg_l2_weight))
        print('Orthogonality loss with weight: {0}'.format(orthog_loss_weight))


    return



# Function: Model Train Setting
def model_train(model, dataloader, device, optimizer, class_specific=False, coefs=None, subtractive_margin=True, use_ortho_loss=False):

    assert optimizer is not None, "If the model is in training mode, you should provide an optimizer."
    model.train()

    return run_model(model=model, dataloader=dataloader, mode="train", device=device, optimizer=optimizer, class_specific=class_specific, coefs=coefs, subtractive_margin=subtractive_margin, use_ortho_loss=use_ortho_loss)



# Function: Model Validation Setting
def model_validation(model, dataloader, device, class_specific=False, subtractive_margin=True):

    model.eval()

    return run_model(model=model, dataloader=dataloader, mode="validation", device=device, optimizer=None, class_specific=class_specific, subtractive_margin=subtractive_margin)



# Function: Model Test Setting
def model_test(model, dataloader, device, class_specific=False, subtractive_margin=True):
    model.eval()

    return run_model(model=model, dataloader=dataloader, mode="test", device=device, optimizer=None, class_specific=class_specific, subtractive_margin=subtractive_margin)



# Function: Prepare layers for warm-only training setting
def last_only(model, last_layer_fixed=True):
    for p in model.features.parameters():
        p.requires_grad = False

    for p in model.add_on_layers.parameters():
        p.requires_grad = False

    model.prototype_vectors.requires_grad = False

    for p in model.conv_offset.parameters():
        p.requires_grad = False

    for p in model.last_layer.parameters():
        p.requires_grad = not last_layer_fixed



# Function: Prepare layers for warm-only training setting
def warm_only(model, last_layer_fixed=True):
    for p in model.features.parameters():
        p.requires_grad = False

    for p in model.add_on_layers.parameters():
        p.requires_grad = True

    model.prototype_vectors.requires_grad = True

    for p in model.conv_offset.parameters():
        p.requires_grad = False

    for p in model.last_layer.parameters():
        p.requires_grad = not last_layer_fixed



# Function: Prepare layers for warm-pre-offset training setting
def warm_pre_offset(model, last_layer_fixed=True):
    for p in model.features.parameters():
        p.requires_grad = True

    for p in model.add_on_layers.parameters():
        p.requires_grad = True

    model.prototype_vectors.requires_grad = True

    for p in model.conv_offset.parameters():
        p.requires_grad = False

    for p in model.last_layer.parameters():
        p.requires_grad = not last_layer_fixed



# Function: Prepare layers for joint training setting
def joint(model, last_layer_fixed=True):
    for p in model.features.parameters():
        p.requires_grad = True

    for p in model.add_on_layers.parameters():
        p.requires_grad = True

    model.prototype_vectors.requires_grad = True

    for p in model.conv_offset.parameters():
        p.requires_grad = True

    for p in model.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
