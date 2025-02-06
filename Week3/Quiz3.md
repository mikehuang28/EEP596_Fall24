# Quiz 3

## What are the limitations of multi-layer perceptron when taking images as inputs and how do CNNs address them?

limitations of MLP:

1. the inputs images have to flatten to 1d vector, large number of input neuron and parameters
2. flatten to 1d vectors can't maintain the images' spatial information
3. MLP tends to overfit the model because of huge numbers of parameters

how CNN address the limitations:

1. weight sharing, which leads to fewer parameters
2. 2d kernel and slide through the input image, preserve spatial information

## With the input image of shape 28 x 28 x 1 (H x W x Channels), what is the shape of the output after going through a convolutional layer with kernel size = 3, stride = 3, and zero padding = 1?

28-3+1*2=27

27/3+1=10

c_out=1

therefore, the output size=10x10x1

## What are two types of pooling layers that are commonly used to contract the output of the convolutional layer? For each type briefly explain its pooling operation.

max pooling: for each region, select the max value of the region, emphasizes strong features

avg pooling: for each region, computes the mean value among the region, smooths and reduces noise

## What are three classical CNNs that were introduced in the lecture? For each network, list few specifics about its architecture that distinguished each network from one another.

LeNet: 5 layers, input size=32 for MNIST, output channel=10, activation functions=tanh and sigmoid, no dropout

AlexNet: 8 layers with 5 conv and 3 fc, input size=224 for imagenet, output channel-1000, activation function=ReLU,  number of channels in conv layers is considered by GPU capacity, dropout used

VGG16: 16 layers, uniform 3x3 filters, activation function=ReLU, dropout used

## Upload a Python code snippet that defines a neural network model class "myCNNModel" with the following specifications.

- Convolution layer 1 = input channels = 1, output channels = 16, kernel size 3, stride = 1, padding = 0, ReLU activation.
- Max pool layer 1 = kernel size 2
- Convolution layer 2 = input channels = 16, output channels = 32, kernel size 3, stride = 1, padding = 0, ReLU activation.
- Max pool layer 2 = kernel size 2
- Fully Connected layer 1 = 800 neurons
- Output layer = 10 neurons

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class myCNNModel(nn.Module):
    def __init__(self):
        super(myCNNModel, self).__init__()

        # first layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # second layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # fully connected layer
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out
```