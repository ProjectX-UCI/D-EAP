# RESOLVES CONFUSING ERROR
import os, sys
from regularizers.functions import Regularizers
from utils.data_utils import load_data
from utils.training_utils import training_loop,package_model_components
from model import Net

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python train_models.py <lambda_reg>")
        raise SyntaxError
    else:
        # get lambda_reg from arguments
        lambda_reg = sys.argv[1]
    
    # assemble regularizer functions to be compared as array
    regularizer_funcs = [Regularizers.none,Regularizers.l1,Regularizers.l2,Regularizers.static_l0]
    regularizer_labels = ["None","L1","L2","Static L0"]

    # package models, regularizers, optimizers and labels to streamline iteration
    model_components = package_model_components(Net,regularizer_funcs,regularizer_labels)

    print("= [Loading data]")
    testloader,trainloader = load_data()

    print('= [Training]')
    for epoch in range(1):

        print('== [Epoch: %s]' % epoch)

        for i, (inputs,labels) in enumerate(trainloader, 0):

            # limit training time for debugging purposes
            if i > 5:
                break
            
            # iterate through all regularizer functions
            for model_package in model_components:

                # run a single training loop
                loss = training_loop(model_package,lambda_reg,inputs,labels)
            
            print(i)

    # save parameters
    # for model_package in model_components:
    #     model_parameters = model_package["model"].state_dict()
    #     model_path = f'./models/{model_package["label"]}.pt'

    #     torch.save(model_parameters, model_path)