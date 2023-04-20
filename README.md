# DeepLossDive

## A Deep Dive into PyTorch Classification Loss Functions
This GitHub repository contains PyTorch classification loss functions, Jupyter notebooks, and documentation for researchers and machine learning enthusiasts interested in deep learning and PyTorch.

### Overview
This repository contains a collection of PyTorch classification loss functions implemented using PyTorch's neural network library. The loss functions are implemented in a modular and easy-to-use manner, allowing users to quickly integrate them into their projects. The code is documented extensively, and Jupyter notebooks are provided to demonstrate the usage of each loss function. Additionally, the repository includes a detailed README file that provides an overview of the project and instructions for running the code.

### Contents
This repository contains the following:

<<...>>

### Requirements
To run the code in this repository, you will need to have the following installed:

Python 3.6 or higher
PyTorch
Jupyter Notebook
Google Collab

## Classification Losses in PyTorch Vision

PyTorch Vision provides several loss functions for classification tasks. These include:

### Cross-Entropy Loss
This loss measures the dissimilarity between the predicted probability distribution and the true distribution. The formula for cross-entropy loss is: $$L = -\sum_{i} y_i \log(p_i)$$ where $y_i$ is the true label and $p_i$ is the predicted probability for class $i$.

Here's an example of using cross-entropy loss in PyTorch:

```python
import torch
import torch.nn.functional as F

# define the true labels
y_true = torch.tensor([0, 1, 2])

# define the predicted probabilities
y_pred = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])

# compute the cross-entropy loss
loss = F.cross_entropy(y_pred, y_true)
print(loss) 
```

## Negative Log-Likelihood Loss
This loss is similar to cross-entropy loss, but it takes as input the log-probabilities instead of probabilities. The formula for negative log-likelihood loss is: $$L = − \log(p_y)$$
 where $p_y$ is the predicted probability for the true class.
 
Here’s an example of using negative log-likelihood loss in PyTorch:

```
import torch
import torch.nn.functional as F

# define the true labels
y_true = torch.tensor([0, 1, 2])

# define the predicted log-probabilities
y_pred = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]))

# compute the negative log-likelihood loss
loss = F.nll_loss(y_pred, y_true)
print(loss)
```

## Hinge Loss
This loss is used for binary classification and is defined as: $$L = max(0, 1 − t . y)$$
 where $t$ is the true label (+1 or -1) and $y$ is the predicted score.
Here’s an example of using hinge loss in PyTorch:

```
import torch
from torch.nn import MarginRankingLoss

# define the true labels (as +1 or -1)
y_true = torch.tensor([1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

# define the predicted scores
y_pred = torch.tensor([4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00,-4.0318e+00])

# compute the hinge loss
loss_fn = MarginRankingLoss(margin=1)
loss = loss_fn(y_pred * y_true, torch.zeros_like(y_pred), y_true)
print(loss)
```

## Binary Cross-Entropy Loss
This loss is used for binary classification tasks and is defined as: $$L = -\sum_{i} y_i \log(p_i) + (1-y_i) \log(1-p_i)$$ where $y_i$ is the true label and $p_i$ is the predicted probability for class $i$.

Here's an example of using binary cross-entropy loss in PyTorch:

```python
import torch
import torch.nn.functional as F

# define the true labels
y_true = torch.tensor([[1., 0.], [0., 1.]])

# define the predicted probabilities
y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7]])

# compute the binary cross-entropy loss
loss = F.binary_cross_entropy(y_pred, y_true)
print(loss)
```

## Kullback-Leibler Divergence Loss in PyTorch

Kullback-Leibler divergence (KL divergence) is a measure of how one probability distribution differs from another. In PyTorch, the `KLDivLoss` class can be used to compute the KL divergence loss between the input and target distributions.

### Formula

For tensors of the same shape `$y_pred$` and `$y_true$`, where `$y_pred$` is the input and `$y_true$` is the target, we define the pointwise KL-divergence as:

$$L(y_pred, y_true) = y_true * (log(y_true) - log(y_pred))$$


To avoid underflow issues when computing this quantity, this loss expects the argument `input` in the log-space. The argument `target` may also be provided in the log-space if `$log_target=True$`.

### Example

Here is an example of how to use `KLDivLoss` in PyTorch:

```python
import torch.nn.functional as F
kl_loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
```

....To do....


### Contributing
Contributions to this repository are welcome. If you find a bug or have an idea for a new feature, feel free to open an issue or submit a pull request.

### Acknowledgments
This project was inspired by the PyTorch documentation and the open-source community of PyTorch users.

Datasets: https://pytorch.org/vision/stable/datasets.html

Loss Functions: https://neptune.ai/blog/pytorch-loss-functions
.
.
.

