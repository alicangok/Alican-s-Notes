Core Ideas:

- Attention is all you need. You don't need the propagating vector used in RNNs.
- In RNNs, the entire sentence-to-sentence body is 1 sample. We need a long back-propagation operation all the way from the last word of the target to the first word of the source.
    - Each sentence is one sample
- In contrast, for the transformer model, the output of 1 token (word) is a sample in itself. Each step is an independent back prop. Since the path length is 1, there is no loss of information during backprop.
    - Each step is one sample

The positional encoding like a continuous version of binary encoding, using a series of sine waves of increasing frequency. It boosts performance by giving the encoders a sense of the position.

Within the transformer, there are 3 places of attention:
1) Input attention
2) Output attention
3) Combining attention: This is the interesting step

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
V: Value from source
K: Key from source
Q: Query from target

The combining attention, takes the Keys and Values from output of the input encoder. Think of Values as interesting things, attributes belonging to the input, and the Keys as vectors that achieve the addressing.

Advantages of Transformers over RNN models:
- better capturing of long term dependencies
- no gradient vanishing / explosion
- requires fewer training steps (due to huge RNN unrolling)
- parallelizable training (no recurrence)



Question for self: How are non-AR transformer models gonna help us? Our output is a single token, no?


Useful resource used: [Yannic Kilcher's video](https://www.youtube.com/watch?v=iDulhoQ2pro)