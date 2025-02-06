# Quiz 4

## What are the limitations of using the direct approach (e.g., fully connected network) for sequence modeling? How do RNNs address them?

Limitations:

1. Fully connected networks have no memory about the past inputs. They cannot capture temperal dependencies.
2. Fully connedted networks deal with fixed inputs and outputs.

How do RNNs address the limitations:

1. RNNs use recurrent connections, where the output of each step depends on previous steps. They create outputs based on the memory of previous inputs.
2. RNNs can process inputs of varying lengths by iterating over each element in a sequence, making them suited for sequence modeling.

## Which RNN types (Lecture slide 46) will you choose for each application (Lecture slide 31) that we discussed at the beginning of the lecture ?

speech recognition, many to one

music generation, one to many

sentiment classification, many to one

dna sequence analysis, many to one

machine translation, many to many

video activity recognition, many to one

name entity recognition, many to many

![image.png](image.png)

## What are some possible solutions to address vanishing or exploding gradients in RNNs?

long short-term memory (LSTM)

gated recurrent unit (GRU)

smaller sequence

gradient clipping

BN

## What are the functions of embedding and decoder in RNN? For each component, briefly explain in a sentence.

The embedding layer converts discrete input tokens (such as words or characters) into continuous vector representations, capturing semantic relationships and reducing dimensionality, which helps the RNN process the data more effectively.

The decoder takes the hidden states from the RNN (usually from the last time step or a series of time steps) and generates the output sequence (e.g., translated words in machine translation or predicted next words), using these hidden states to inform its predictions based on the context learned from the input sequence.

## Upload a Python code snippet that defines a neural network model class "myRNNModel" with the following specifications.

- **Embedding layer:** # of embeddings: 10, embedding dimension: 30
- **RNN Cell:** input_size = 30, hidden_size = 128, num_layers = 1, nonlinearity = 'tanh'
- **Decoder:** input dimension = 128, output dimension = 10

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class myRNNModel(nn.Module):
    def __init__(self):
        super(myRNNModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=30)
        self.rnn = nn.RNN(input_size=30, hidden_size=128, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.decoder = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # x should be of shape (batch_size, sequence_length)

        # Pass through embedding layer
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Pass through RNN layer
        rnn_out, hidden = self.rnn(embedded)  # rnn_out shape: (batch_size, sequence_length, hidden_size)

        # Use the last hidden state for decoding
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_size)

        # Pass through decoder
        out = self.decoder(last_hidden)  # Shape: (batch_size, output_dim)

        return out
```