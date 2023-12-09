import torch.optim as optim
import torch.nn as nn
import torch, os
import time, sys

sys.path.append('../')

criterion = nn.CrossEntropyLoss()

def package_model_components(model_class,list_of_regularizers):
    model_components = []

    for regularizer in list_of_regularizers:
        model = model_class()
        compiled_regularizer = regularizer(model=model, lambda_reg=10**-2)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        model_component = {
            "model":model,
            "regularizer":compiled_regularizer,
            "optimizer":optimizer
        }

        model_components.append(model_component)
    
    return model_components

def training_loop(model_package,inputs,labels) -> "loss":
    model = model_package["model"]
    regularizer = model_package["regularizer"]
    optimizer = model_package["optimizer"]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # REGULARIZATION
    loss = regularizer.regularized_all_param(reg_loss_function=loss)

    loss.backward()
    optimizer.step()

    return loss

def calculate_regularizer_metrics(model_components,inputs,labels):

    loss_across_regularizers = []
    latency_across_regularizers = []

    for model_package in model_components:
        
        #time each training loop to get latency
        start = time.time()
        PATH = f"./models/{model_package['regularizer']}.pt"
        if os.path.exists(PATH):
            print("Loading model from disk")
            model_package["model"].load_state_dict(torch.load(PATH))
        loss = training_loop(model_package,inputs,labels)
        torch.save(model_package["model"].state_dict(), PATH)
        end = time.time()

        loss_across_regularizers.append(loss.item())
        latency_across_regularizers.append(end - start)

    return loss_across_regularizers,latency_across_regularizers