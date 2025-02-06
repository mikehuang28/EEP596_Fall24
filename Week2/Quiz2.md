# Quiz 2

## What are some pros and cons of using Stochastic Gradient Descent (SGD) over the Batch Gradient Descent (GD)?

pros: consistently converge to the minimum, possibly take shortcut to the minimum

cons: not useful when close to minimum, hard to parallelize because passing samples little by little

## What are the advantages of having a "variable" learning rate over the "static" learning rate during optimization?

It has a better chance to reach the minimum. sometimes we are near the minimum point but miss it because the learning rate is a bit too large. There is not enough room to go.

## What are some methods we can use to alleviate overfitting? For each method briefly explain in a sentence the mechanism the method uses to alleviate overfitting.

1. more training data: can form a smoother line to fit all the data rather than the jumpy one
2. regularization: add penalties for large weights in the model to make the model to find simplier solutions
3. dropout: randomly dropping out neurons forces the model to learn meaningful features
4. initialization: prevent the model from getting stuck in suboptimal solutions or learning imbalances
5. early stopping: very easy but effective
6. BN: extreme gradient prevention

## Which components during neural network training are required to be controlled/normalized to attempt to avoid vanishing/exploding gradient effects?

1. do the data splitting and normalize all datasets, including training sets, validation sets and testing sets (normalize all datasets very very important)
2. weight Initialization plays essential roles in preventing exploding/vanishing gradients, leading to faster convergence
3. network initialization help keep the scale of the weights balanced, preventing gradients from shrinking too small or growing too large
4. batch normalization to normalize the activations between layers help maintain a stable gradient flow
5. hyperparameter tuning such as selecting proper learning rate ensures gradients update weights properly

## Upload a Python code snippet that defines a neural network model class "myModel" with the following specifications.
- Input layer = 10 neurons
- First hidden layer = 32 neurons, Sigmoid activation
- Second hidden layer = 16 neurons, Sigmoid activation, Dropout with 30% probability (applied after sigmoid activation)
- Output layer = 5 neurons, Softmax activation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

        # input layer 10 neurons, first hidden layer 32 neurons
        self.fc1 = nn.Linear(10, 32)
        self.sigmoid1 = nn.Sigmoid() # sigmoid

        # second hidden layer
        self.fc2 = nn.Linear(32, 16)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3) # dropout

        # output layer
        self.fc3 = nn.Linear(16, 5)

    def forward(self, x):
        x = self.sigmoid1(self.fc1(x))
        x = self.dropout(self.sigmoid2(self.fc2(x))) # dropout after sigmoid
        out = F.softmax(self.fc3(x), dim=1) # output layer with softmax
        return out
```