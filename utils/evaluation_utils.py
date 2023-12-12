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
    def __init__(self, model, model_path, data_loader, device='cpu'):
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.data_loader = data_loader
        self.device = device

    def evaluate(self, measure_latency = True) -> (float,float):
        '''
        Evaluator.evaluate() returns the accuracy of the model
        '''
        correct = 0
        total = 0

        iterations = 0
        latency = None

        if measure_latency:
            start_time = time.time()

        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                iterations += 1

        accuracy = correct / total

        if measure_latency:
            end_time = time.time()
            latency = (end_time - start_time) / iterations
            latency *= 1000
        
        return accuracy,latency

    @profile
    def memory_usage(self) -> None:
        # initialize evaluation Object in main and run
        # python -m memory_profiler main_experiment.py
        for inputs, _ in self.data_loader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)