### Building blocks:
- loss function (for single example)
- optimization criterion, cost function (for dataset)
- optimization strategy (SGD, etc.)

### Gradient Descent:
- one pass through all training examples: epoch
- apply chain rule to backprop error to features
- GD: calculate gradients using all training examples
- SGD: speed up by using smaller batches
- adagrad: scale LR for each parameter using history
- momentum: use exponentially weighted average of gradients (~0.9)
    - stops zigzags, by averaging gradients history
- adam: combines momentum and RMSprop (Root Mean Square Propagation)