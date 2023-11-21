# RESOLVES CONFUSING ERROR
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from regularizers.regularizer import *
from utils.data_utils import load_data
from utils.training_utils import calculate_regularizer_metrics,package_model_components
from models.model import Net
from utils.evaluation_utils import plotDF

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

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

        print(' == [Epoch: %s]' % epoch)

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

    print('= [Visualizing Results]')

    # print("loss_across_epochs",loss_across_epochs)
    # print("latency_across_epochs",latency_across_epochs)

    plotDF(loss_across_epochs,columns=columns,title="Loss Across Epochs",ylabel="loss",plot_figure=False,save_figure=True)
    plotDF(latency_across_epochs,columns=columns,title="Latency Across Epochs",ylabel="latency",plot_figure=False,save_figure=True)