### Neural Networks, MLP, Feed-forward nets

- a neural net is a nested function, like y=f3(f2(f1(x)))
- each layer is something like: f(z) = g(Wz+b)
    - g: activation function
        - TanH (-1 to 1)
        - ReLU
        - sigmoid / softmax: final layer for classification
    - W: weight matrix, b: bias

### Deep Learning

- two big challenges
    - exploding gradient: easy to deal with
        - gradient clipping
        - L1, L2 regularization
    - vanishing gradient: intractable for decades
        - ReLU suffers less than TanH
            - many TanH layers, with each a gradient between (0,1), the chain rule will make the earlier layers train very slowly
        - LSTM
        - skip connections in residual neural networks

### CNN

- can reduce number of parameters drastically
    - imagine 32x32 image: regular MLP would need 1k x 1k = 1M parameters per layer: difficult to optimize, compute even
- used in image processing

### RNN / GRU / LSTM

- to label, classify or generate sequences
- used text and speech processing
- state: seaves as memory
- each unit in each layer has two inputs
    - vector of outputs from previous layer
    - vector of states from same layer, previous time step
- typical activation for state calculation: tanh
- typical activation for output calculation: softmax
- two major problems:
    - both tanh and softmax suffer from vanishing gradient
    - long term dependencies are often "forgetten"
- most effective in practice are gated RNNs:
    - GRU: update and reset gates
        - gated unit takes an input and stores it, depending on update gate output (if gate output is 0, it keeps the previous state, equivalent to identity)
        - thus gradient vanishing is solved
    - LSTM: two gates: forget & input
        - technically: output gate as well