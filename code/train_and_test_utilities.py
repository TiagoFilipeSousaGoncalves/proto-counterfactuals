# Source: https://github.com/cfchen-duke/ProtoPNet/blob/master/train_and_test.py
# Imports
import time
from tqdm import tqdm

# PyTorch Imports
import torch



# Function: List of Distances
def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)



# Function: Train or Test Phase
def _train_or_test(model, dataloader, is_train, device, optimizer=None, class_specific=True, use_l1_mask=True, coefs=None):


    # Check if we are in training mode
    if is_train:
        assert optimizer is not None, "If the model is in training mode, you should provide an optimizer."


    # Some control variables
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0


    # Separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0


    # Iterate through the dataloader
    for _, (image, label) in enumerate(tqdm(dataloader)):
        
        # Put data into device
        # image = image.cuda()
        # label = label.cuda()
        image, label = image.to(device), label.to(device)


        # Note: torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        

        # Get model's outputs
        with grad_req:
            
            # We pass image by the model    
            output, min_distances = model(image)

            # We first compute the CrossEntropy Loss
            cross_entropy = torch.nn.functional.cross_entropy(output, label)

            # Check this condition
            if class_specific:
                # max_dist = (model.module.prototype_shape[1] * model.module.prototype_shape[2] * model.module.prototype_shape[3])
                max_dist = (model.prototype_shape[1] * model.prototype_shape[2] * model.prototype_shape[3])


                # Note: prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # Calculate cluster cost
                # prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                # prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).to(device)
                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,label]).to(device)
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


            # Evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += label.size(0)
            n_correct += (predicted == label).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()


        # Compute final loss value
        if is_train:
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


            # Compute gradients and do backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # TODO: Is this necessary?
        del image
        del label
        del output
        del predicted
        del min_distances


    # Some log prints
    print('\ttime: \t{0}'.format(time.time() -  start))
    print('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    print('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))


    if class_specific:
        print('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        print('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))

    print('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    # print('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    print('\tl1: \t\t{0}'.format(model.last_layer.weight.norm(p=1).item()))


    # Get prototypes
    # p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()


    # Compute prototype average pair distance
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    print('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))



    return n_correct / n_examples



# Function: Train
def train(model, dataloader, device, optimizer, class_specific=False, coefs=None):

    # TODO: Erase uppon review
    # assert(optimizer is not None)

    # If model is not in training mode, change it to training mode
    if not model.training:
        print('\ttrain')
        model.train()


    return _train_or_test(model=model, dataloader=dataloader, device=device, is_train=True, optimizer=optimizer, class_specific=class_specific, coefs=coefs)



# Function: Test
def test(model, dataloader, device, class_specific=False):

    # If model in train training mode, change it to evaluation mode
    if model.training:
        print('\ttest')
        model.eval()


    return _train_or_test(model=model, dataloader=dataloader, device=device, is_train=False, optimizer=None, class_specific=class_specific)



# Function: 
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
    
    print('\tlast layer')



# Function:
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
    
    print('\twarm')



# Function:
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
    
    print('\tjoint')
