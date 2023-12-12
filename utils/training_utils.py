import torch.optim as optim
import torch.nn as nn
import torch, os
import time, sys

sys.path.append('../')

criterion = nn.CrossEntropyLoss()

# move to model_utils/
def package_model_components(model_class,regularizer_funcs,labels):
    model_components = []

    for regularizer_func,label in zip(regularizer_funcs,labels):
        model = model_class()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        model_component = {
            "model":model,
            "regularizer_func":regularizer_func,
            "optimizer":optimizer,
            "label":label
        }

        model_components.append(model_component)
    
    return model_components

def training_loop(model_package,lambda_reg,inputs,labels) -> "loss":
    model = model_package["model"]
    regularizer_func = model_package["regularizer_func"]
    optimizer = model_package["optimizer"]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    # loss = criterion(outputs, labels)

    # # REGULARIZATION
    # loss = regularizer.regularized_all_param(reg_loss_function=loss)

    # calculate regularization values for all nodes
    regularization_values = [regularizer_func(params) for type, params in model.named_parameters() if type.endswith('weight')]
    
    # calculate original loss + weighted regularization value
    loss = criterion(outputs, labels) + lambda_reg * sum(regularization_values)

    loss.backward()
    optimizer.step()

    return loss