# Dynamic Energy-Aware Pruning

In this project, we implement a novel regularization term to differentiably calculate the l0 norm of a parameter vector, and use this to sparsify a neural network model in novel ways. We compare the performance of this regularizer to other standard regularization terms (L1, L2) in terms of accuracy, latency and sparsity at the time of inference.

We compare these regularization terms by training multiple ResNet models on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

This project was created for [ProjectX's 2023 cycle](https://www.uoft.ai/projectx-2023), a research competition hosted by the University of Toronto for computational efficiency in machine learning.

### Requirements
- memory-profiler
- torch
- torchvision
- matplotlib
- numpy
- pickle
- scikit-learn

### Instructions for Use

1. TRAIN MODELS: "python ./train_models <test_label> <lambda_reg>"
  - test_label (string): name of training context folder that will be created to store trained model parameters
  - lambda_reg (float): modulation value to scale the impact of regularization terms
2. EVALUATE MODELS "python ./evaluate_models <test_label>"

### Credits

Undergraduate students at University of California, Irvine
- Redford Hudson, Data Science
- Aparajita Bandopadhyay, Mathematics
- Alex Wong, Computer Science
- Souzen Khan, Computer Science
- Azra Zahin, Computer Science
