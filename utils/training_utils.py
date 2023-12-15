import torch.optim as optim
import torch.nn as nn
import sys

sys.path.append('../')

# move to model_utils/
def package_model_components(model_class,regularizer_funcs,labels):
    """Assembles a list of dictionaries for every regularizer to analyze, with corresponding models, optimizers and labels included in the dictionary

    Args:
        model_class (nn.Module): base neural network model class to be initialized for each regularizer
        regularizer_funcs (list(function)): list of regularizers
        labels (list(string)): list of labels for regularizers

    Returns:
        list(dict): list of dictionaries containing necessary elements of regularizers for training
    """
    model_components = []

    for regularizer_func,label in zip(regularizer_funcs,labels):

        # initialize model
        model = model_class()

        # initialize optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # package components in dictionary
        model_component = {
            "model":model,
            "regularizer_func":regularizer_func,
            "optimizer":optimizer,
            "label":label
        }

        model_components.append(model_component)
    
    return model_components

# baseline loss function
criterion = nn.CrossEntropyLoss()

def training_loop(model_package,lambda_reg,inputs,labels) -> "loss":
    """Runs a single training loop for a single model

    Args:
        model_package (list(dict)): list of dictionaries containing necessary elements of regularizers for training
        lambda_reg (float): modulation value to scale the impact of regularization terms
        inputs (torch.Tensor): data to be predicted upon
        labels (torch.Tensor): actual outputs

    Returns:
        loss (float): loss value
    """

    model = model_package["model"]
    regularizer_func = model_package["regularizer_func"]
    optimizer = model_package["optimizer"]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)

    # calculate regularization values for all nodes
    regularization_values = [regularizer_func(params) for type, params in model.named_parameters() if type.endswith('weight')]
    
    # calculate baseline loss + modulated regularization value
    loss = criterion(outputs, labels) + lambda_reg * sum(regularization_values)

    # calculate gradients with respect to loss
    loss.backward()

    # apply gradients to parameters
    optimizer.step()

    # return loss value for analysis
    return loss.item()