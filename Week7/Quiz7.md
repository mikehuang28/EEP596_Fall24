# Quiz 7

## How does the addition of attention layer help seq2seq model to better encode long-term context?

The attention layer helps seq2seq models encode long-term context by allowing the decoder to dynamically focus on relevant parts of the input sequence through weighted access to all encoder hidden states, rather than relying on a fixed-length context vector. This improves the model's ability to handle long sequences and ensures that important details are preserved for generating accurate outputs.

more efficient in capturing long term content 

## In the context of Seq2Seq model, what was the core contribution of the paper "Attention is all you need"?

The paper introduces transformer, which eliminated the reliance on recurrence and convolution, using only a self-attention mechanism to capture dependencies between input and output sequences. This innovation enabled parallelization during training, improved scalability, and set new benchmarks in sequence modeling tasks by efficiently handling long-range dependencies.

## What are the intuitions behind query, key and value matrices in self-attention layer?

The intuitions behind **query**, **key**, and **value** matrices in a self-attention layer are inspired by information retrieval concepts:

1. **Query (Q):**
    - Represents the current element (or token) seeking context or information.
    - Intuitively, it asks, "What information do I need to focus on in the sequence?"
2. **Key (K):**
    - Encodes all elements in the sequence as potential sources of information.
    - It defines "What information do I have to offer?" for each token.
3. **Value (V):**
    - Represents the actual information contained in each element.
    - If a token is deemed important (via the query-key matching), its value contributes proportionally to the attention output.

The **dot product of query and key** determines how relevant each key is to the query, and the resulting attention weights are used to aggregate the values into a weighted sum, allowing the model to focus on the most relevant parts of the sequence.

## What is the benefit of having multi-headed attention vs a single headed attention layer?

Multi-headed attention allows the model to focus on different aspects of the input sequence simultaneously, such as syntax and semantics, capturing more diverse relationships compared to single-headed attention. It also improves representational power and efficiency by splitting computations across multiple lower-dimensional heads.

## In the context of sequence modeling, what are some benefits of using transformer architecture as opposed to RNNs?

The Transformer architecture offers several benefits over RNNs for sequence modeling:

1. **Parallelization:** Transformers process entire sequences simultaneously, leveraging self-attention, while RNNs process tokens sequentially, making Transformers significantly faster, especially for long sequences.
2. **Long-Range Dependency Modeling:** Self-attention allows Transformers to capture global context efficiently, whereas RNNs struggle with vanishing gradients and lose information over long sequences.
3. **Scalability:** Transformers scale better with modern hardware (e.g., GPUs/TPUs) due to their matrix-based operations, while RNNs are inherently limited by their sequential nature.
4. **Better Performance:** Transformers have consistently outperformed RNNs on various benchmarks, achieving state-of-the-art results in tasks like translation, summarization, and language modeling.