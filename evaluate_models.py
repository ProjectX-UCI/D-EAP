from utils.evaluation_utils import plotDF, Evaluator
import torch
from regularizers.regularizer import *
from utils.training_utils import package_model_components
from model import Net
from utils.data_utils import load_data

if __name__ == '__main__':
    
    list_of_regularizers = [NullRegularizer,L1Regularizer,L2Regularizer]
    labels = ["Null","L1","L2"]
    model_components = package_model_components(Net,list_of_regularizers,labels)

    print("= [Loading data]")
    testloader,trainloader = load_data()

    # INFERENCE MODULE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_package in model_components:
        model_path = f'./models/{model_package["label"]}.pt'
        inference_evaluator = Evaluator(model_package['model'], model_path, testloader, device)

        print(inference_evaluator.evaluate())
        print(inference_evaluator.latency())
        inference_evaluator.memory_usage()
    # print('= [Visualizing Results]')

    # print("loss_across_epochs",loss_across_epochs)
    # print("latency_across_epochs",latency_across_epochs)

    # plotDF(loss_across_epochs,columns=columns,title="Loss Across Epochs",ylabel="loss",plot_figure=False,save_figure=True)
    # plotDF(latency_across_epochs,columns=columns,title="Latency Across Epochs",ylabel="latency",plot_figure=False,save_figure=True)