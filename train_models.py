import os, sys, torch
from utils.file_utils import *
from regularizers.functions import Regularizers
from utils.data_utils import load_data
from utils.training_utils import training_loop,package_model_components
from model import Net

# WHAT IS THIS FOR?
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(lambda_reg,test_label):
    # assemble regularizer functions to be compared as array
    regularizer_funcs = [Regularizers.none,Regularizers.l1,Regularizers.l2,Regularizers.static_l0]
    regularizer_labels = ["None","L1","L2","Static L0"]

    # package models, regularizers, optimizers and labels to streamline iteration
    model_components = package_model_components(Net,regularizer_funcs,regularizer_labels)

    print("= [Loading data]")
    _,trainloader = load_data()

    loss_output = []

    print('= [Training]')
    for epoch in range(1):

        print('== [Epoch: %s]' % epoch)

        for i, (inputs,labels) in enumerate(trainloader, 0):
            loss_output_for_epoch = []

            # limit training time for debugging purposes
            if i > 5:
                break
            
            # iterate through all regularizer functions
            for model_package in model_components:

                # run a single training loop
                loss = training_loop(model_package,lambda_reg,inputs,labels)
                loss_output_for_epoch.append(loss)
            
            print(i)

            # eventually, move outside loop (datum -> epoch)
            loss_output.append(loss_output_for_epoch)

    # create folder to store loss and model parameters for particular training run
    create_folder(folder_path)
    
    # save loss
    path = f"./models/{test_label}/loss_output.csv"
    store_csv(loss_output,regularizer_labels,path)

    # save parameters
    for model_package in model_components:
        model_parameters = model_package["model"].state_dict()
        model_path = f'./models/{test_label}/{model_package["label"]}.pt'

        torch.save(model_parameters, model_path)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python train_models.py <lambda_reg>")
        raise SyntaxError
    else:
        # get lambda_reg from arguments
        try:
            lambda_reg = float(sys.argv[1])
        except:
            raise TypeError
        test_label = sys.argv[2]
    
    # create folder to store files for this particular training run
    folder_path = f"./models/{test_label}"
    if check_folder_exists(folder_path):
        raise FileExistsError

    main(lambda_reg,test_label)