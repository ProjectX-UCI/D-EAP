# RESOLVES CONFUSING ERROR
import os, torch
from regularizers.regularizer import *
from utils.data_utils import load_data
from utils.training_utils import calculate_regularizer_metrics,package_model_components
from model import Net
from utils.evaluation_utils import plotDF, Evaluator

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    print("= [Loading data]")

    testloader,trainloader = load_data()

    loss_across_epochs = []
    latency_across_epochs = []
    list_of_regularizers = [NullRegularizer,L1Regularizer]
    columns = ["NullRegularizer","L1Regularizer"]

    model_components = package_model_components(Net,list_of_regularizers)

    print('= [Training]')

    for epoch in range(1):  # loop over the dataset multiple times

        print('== [Epoch: %s]' % epoch)

        for i, (inputs,labels) in enumerate(trainloader, 0):
            # limit training time for debugging purposes
            if i > 5:
                break

            #since we are not interested in inference, loss and latency can be calculated in one training loop
            loss_across_regularizers,latency_across_regularizers = calculate_regularizer_metrics(model_components,inputs,labels)
            loss_across_epochs.append(loss_across_regularizers)
            latency_across_epochs.append(latency_across_regularizers)
            
            # calculate_FLOPS()

    # INFERENCE MODULE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in model_components:
        model_path = f'./models/{model["regularizer"]}.pt'
        inference_evaluator = Evaluator(model['model'], model_path, testloader, device)

        print(inference_evaluator.evaluate())
        print(inference_evaluator.latency())
        inference_evaluator.memory_usage()
    print('= [Visualizing Results]')

    # print("loss_across_epochs",loss_across_epochs)
    # print("latency_across_epochs",latency_across_epochs)

    plotDF(loss_across_epochs,columns=columns,title="Loss Across Epochs",ylabel="loss",plot_figure=False,save_figure=True)
    plotDF(latency_across_epochs,columns=columns,title="Latency Across Epochs",ylabel="latency",plot_figure=False,save_figure=True)