import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from memory_profiler import profile

def plotDF(df,columns,title,ylabel,plot_figure=False,save_figure=False):
    df = pd.DataFrame(df, columns=columns)
    plot = df.plot(title=title)
    plot.set(xlabel="Time/Iteration", ylabel=ylabel)

    if plot_figure:
        plt.show()
    
    if save_figure:
        plt.savefig("./results/%s.jpg"%title)

class Evaluator:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def evaluate(self, mode = "Validation") -> str:
        '''
        mode: "Validation" or "Test"
        Evaluator.evaluate() returns the accuracy of the model
        '''
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return f'{mode} Accuracy: {accuracy * 100:.2f}%'

    def latency(self, num_iterations = 100) -> str:
        '''
        num_iterations: number of iterations to run inference on
        Evaluator.latency() returns the average latency of the model
        '''
        self.model.eval()
        start_time = time.time()
        for i, (inputs, _) in enumerate(self.data_loader):
            if i >= num_iterations:
                break
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
        end_time = time.time()
        latency = (end_time - start_time) / num_iterations
        return f'Average Latency: {latency * 1000:.2f} ms'

    @profile
    def memory_usage(self) -> None:
        # initialize evaluation Object in main and run
        # python -m memory_profiler main_experiment.py
        self.model.eval()
        for inputs, _ in self.data_loader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)