import torch
import torch.nn as nn

class SparsityCalculator(nn.Module):
    def __init__(self, model):
        super(SparsityCalculator, self).__init__()
        self.model = model
        self.sparsity = None

        # Hook for each parameter in the model
        self.hooks = []
        for param in self.model.parameters():
            hook = param.register_hook(self.compute_sparsity)
            self.hooks.append(hook)

    def forward(self, x):
        _ = self.model(x)  # Forward pass to trigger hooks
        return x

    def compute_sparsity(self, grad):
        non_zeros = torch.sum(grad != 0).item()
        total_params = grad.numel()
        sparsity = 1.0 - (non_zeros / total_params)
        self.sparsity = sparsity

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Example usage:
# Create your model
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define your model layers here

    def forward(self, x):
        # Define the forward pass
        return x

# Instantiate your model
your_model = YourModel()

# Wrap your model with the SparsityCalculator
sparsity_calculator = SparsityCalculator(your_model)

# Use the sparsity_calculator for inference
input_data = torch.randn(1, 10)
output = sparsity_calculator(input_data)

# Access the sparsity value
sparsity_value = sparsity_calculator.sparsity
print(sparsity_value)

# Don't forget to remove hooks when done
sparsity_calculator.remove_hooks()
