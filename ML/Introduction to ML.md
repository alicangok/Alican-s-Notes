ML: Solve practical problem by gathering a dataset and building a statistical model based on it.

- Supervised learning: Labeled examples (y), with feature vectors (x)
    - Labels: either classes or a number (usually)
    - Goal: produce a model that uses features to deduce the label
- Unsupervised learning: No labels, just features
    - Transform the vector into a value to solve a practical problem
    - Clustering / dimensionality reduction
- Semi-supervised learning: Dataset has both labeled and unlabeled examples
    - Hope: unlabeled examples can help the learning algorithm
- Reinforcement learning:
    - Machine perceives state -> execute actions
    - Through rewards, learn a policy (state -> optimal action)
    - Robotics, game playing, resource management

### Supervised Learning:

- First step: Gather data
    - Convert data into feature vector (such as bag of words)
    - Transform labels into numbers
- Apply an algorithm to dataset, to get the model: Training
    - learn parameters that decide the decision boundary
    - solve an optimization problem
- The model works on new data:
    - Assumption: test set has similar distribution to training set