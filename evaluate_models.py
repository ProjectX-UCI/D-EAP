import torch, sys

from utils.file_utils import *
from utils.evaluation_utils import plotDF, Evaluator
from regularizers.functions import Regularizers
from utils.training_utils import package_model_components
from model import Net
from utils.data_utils import load_data

def main(folder_path):

    print("= [Loading data]")
    testloader,_ = load_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # for all regularizers, evaluate accuracy, latency, memory and sparsity
    for file_name in find_files_with_extension(folder_path,'.pt'):
        print(f"= [Evaluating Regularizer: {file_name.split('.pt')[0]}]")

        file_path = f"{folder_path}/{file_name}"
        inference_evaluator = Evaluator(Net(), file_path, testloader, device)

        print(inference_evaluator.evaluate())
        print(inference_evaluator.latency())
        inference_evaluator.memory_usage()

    # plotDF(loss_across_epochs,columns=columns,title="Loss Across Epochs",ylabel="loss",plot_figure=False,save_figure=True)
    # plotDF(latency_across_epochs,columns=columns,title="Latency Across Epochs",ylabel="latency",plot_figure=False,save_figure=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_models.py <test label>")
        raise SyntaxError
    else:
        test_label = sys.argv[1]

    # if folder DNE, break
    folder_path = f"./models/{test_label}"
    if not check_folder_exists(folder_path):
        raise FileNotFoundError

    main(folder_path)