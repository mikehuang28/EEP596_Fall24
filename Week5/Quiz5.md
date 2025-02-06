# Quiz 5

## What is the purpose of cell state in LSTM? Describe in a sentence or two.

The cell state in an LSTM network serves as a long-term memory that helps retain information across time steps, mitigating issues like vanishing gradients. It allows the LSTM to preserve important information over longer sequences by passing relevant data unchanged, while selectively adding or removing information at each step via gates.

cell states: memory stores long-term memory, for prevent extreme gradients, add more interpretability to RNN

## In what situations would you prefer using LSTM, GRU over vanilla RNN? Explain your choices in a sentence.

LSTMs and GRUs are preferable over vanilla RNNs when dealing with long sequences or complex patterns in data, as they address vanishing gradient issues and efficiently capture long-term dependencies, making them well-suited for tasks like language modeling and time-series forecasting.

## Deep-RNN and Bi-directional RNN extensions are not applicable to LSTM, GRU (True/False). Explain your answer.

False.

Both Deep RNNs and Bi-directional RNNs can be applied to LSTMs and GRUs. Deep RNNs involve multiple layers of LSTM or GRU cells to capture complex features at different levels, while Bi-directional RNNs use forward and backward passes to gather information from both past and future contexts, enhancing model performance on tasks like language processing.

## What are the main differences between time-synced Many-to-many RNNs and Encoder-Decoder architecture?

Time-synced Many-to-Many RNNs process input and output sequences of the same length simultaneously, while Encoder-Decoder architectures first encode the entire input sequence into a fixed representation before decoding it into a separate output sequence, often suited for tasks where input and output lengths differ, like machine translation.

## Provide an example problem you can think of where Encoder-Decoder would be a suitable architecture. What would be the input and output sequence for the problem of your choice?

Machine translation. The input sequence would be a sentence in English (e.g., "How are you?"), and the output sequence would be its translation in Chinese (e.g., "你好嗎？").

The Encoder-Decoder framework is effective here because it can handle different input and output lengths, capturing the semantic meaning of the input sentence before generating the translated output.