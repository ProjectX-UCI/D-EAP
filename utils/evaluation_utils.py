import pandas as pd
import matplotlib.pyplot as plt
import torch, time
import subprocess as sp
# from memory_profiler import profile

def plotDF(df,columns,title,ylabel,plot_figure=False,save_figure=False):
    """Plots dataframe containing loss values across time (epochs) for different regularizer-model packages

    Args:
        df (pd.dataframe)
        columns (list(string))
        title (string)
        ylabel (string)
        plot_figure (bool, optional): Defaults to False.
        save_figure (bool, optional): Defaults to False.
    """
    df = pd.DataFrame(df, columns=columns)
    plot = df.plot(title=title)
    plot.set(xlabel="Time/Iteration", ylabel=ylabel)

    if plot_figure:
        plt.show()
    
    if save_figure:
        plt.savefig("./results/%s.jpg"%title)


class Evaluator:
    """Class for evaluating regularizers in terms of various measurement benchmarks
    """

    def __init__(self, model, model_path, data_loader, device='cpu'):
        self.model = model
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.data_loader = data_loader
        self.device = device

    def evaluate(self, measure_latency = True) -> (float,float):
        """Measures accuracy and latency of a model at the time of inference

        Returns:
            (float,float): tuple containing accuracy and latency score respectively
        """
        correct = 0
        total = 0

        iterations = 0
        latency = None

        if measure_latency:
            start_time = time.time()

        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass -> probability of predicting each class
                outputs = self.model(inputs)

                # gets index of highest probability in output vector
                # index of highest probability = predicted label
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                iterations += 1

        accuracy = correct / total

        if measure_latency:
            end_time = time.time()
            latency = (end_time - start_time) / iterations
            latency *= 1000
        return accuracy, latency

    # @profile
    # def memory_usage(self) -> None:
    #     # initialize evaluation Object in main and run
    #     # python -m memory_profiler main_experiment.py
    #     for inputs, _ in self.data_loader:
    #         inputs = inputs.to(self.device)
    #         with torch.no_grad():
    #             outputs = self.model(inputs)
    
    def sparsity(self, threshold = 10e-5) -> float:
        """Measures sparsity of a model, where sparsity is the number of parameters with magnitude coefficients below the given threshold value

        Args:
            threshold (float, optional): Threshold for determining whether a parameter is considered zero or non-zero. Defaults to 10e-5.

        Returns:
            float: sparsity score
        """

        # Count the total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        # Count the number of non-zero parameters
        zero_params = sum(torch.sum(torch.abs(p) < threshold).item() for p in self.model.parameters())

        # Calculate sparsity
        sparsity = (zero_params / total_params)
        return sparsity