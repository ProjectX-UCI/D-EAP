# RESOLVES CONFUSING ERROR
import os, torch
from regularizers.regularizer import *
from utils.data_utils import load_data
from utils.training_utils import training_loop,package_model_components
from model import Net

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':    
    
    list_of_regularizers = [NullRegularizer,L1Regularizer,L2Regularizer]
    labels = ["Null","L1","L2"]
    model_components = package_model_components(Net,list_of_regularizers,labels)

    print("= [Loading data]")
    testloader,trainloader = load_data()

    print('= [Training]')
    for epoch in range(1):

        print('== [Epoch: %s]' % epoch)

        for i, (inputs,labels) in enumerate(trainloader, 0):

            # limit training time for debugging purposes
            if i > 5:
                break
            
            for model_package in model_components:
                model_path = f"./models/{model_package['regularizer']}.pt"
                loss = training_loop(model_package,inputs,labels)
            
            print(i)

    # save parameters
    for model_package in model_components:
        model_parameters = model_package["model"].state_dict()
        model_path = f'./models/{model_package["label"]}.pt'

        torch.save(model_parameters, model_path)