### Imbalanced datasets

- set cost of misclassification higher for minority class
    - or you can oversample (duplicate) examples for a class
- Decision trees / random forest / boosting works well on imbalanced sets

### Combining models

- typical ways: averaging, majority, stacking
- averaging: regression or classification scores
- majority: classification
- stacking: build meta model that takes output of base models as input
    - supervised training
- critical assumption: need uncorrelated models to see improvement
    - same architecture with different hyperparameters will likely produce correlated models

### Training neural networks

- start with a simple model (few layers)
- gradual increase size until model fits training data
- observe validation data: if high variance add regularization
- now if training results suffer, increase size of network

### Advanced regularization

- basic methods: L1, L2 regularization
- dropout: randomly exclude units from computation, 2 main advantages
    - Prevents overfitting
    - Provides a way to combine exponentially many neural network architectures efficiently
    - Scaling necessary: If a unit is retained with probability p during training, the outgoing weights of that unit are multiplied by p at test time
- batch normalization (technically not a regularizer)
    - standardize outputs of each layer before next layer
    - faster and more stable training
- early stopping / revert to checkpoint: to prevent overfitting
- data augmentation: synthetic examples (rotate/zoom/noise)

### Multiple inputs / outputs

- multimodal scenario
- build subnetworks for each input
    - e.g. combine CNN&RNNs for image&text embeddings, then concat
- outputs: label and coordinates
    - after encoder, have two subnetworks for output types
    - cost function combines the two losses (weight is another hyperparameter)

### Transfer Learning

- remove last few layers from an original model
- replace removed layers with new layers suitable to your problem
- freeze the layers from the original model, then train