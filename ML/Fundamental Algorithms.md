### Linear Regression

- Model is a linear combination of inbput features
- The hyperplane in SVM is now kind of the best fit hyperplane
- Can have linear regression in a quadratic function, for example
- model: y = wx + b

### Logistic Regression

- misnomer: the goal is not regression, it's classification
- Sigmoid: $\frac{1}{1+e^{-x}}$
- probability is modeled with Bernoulli RV
- optimization criterion: maximum likelihood
    - product of individual probabilities
    - turns out the solution minimizes cross entropy
        - binary: $−(𝑦*log(𝑝)+(1−𝑦)*log(1−𝑝))$
        - multiclass: minus sum of log(probs) for non-correct labels

### Decision Tree
- various formulations
- a notable one is ID3
    - objective: make such a separation as to minimize the entropy in two sides of the tree

### SVM

Parametric, but kernelized SVM is non-parametric

- goals:
    - wx-b > 1 for +1 labels
    - wx-b < -1 for -1 labels
    - minimize L1(w) (or equivalently L2(w) square), such that the top conditions are satisfied: y(xw-b)-1>0
- Solved through Lagrangian multiplier / dual forms -> convex problem
- If data is not linearly separable, introduce hinge loss:
    - max(0, 1-y(wx-b))
        - if wx on the correct side: loss 0
        - else, loss is proportional to distance to boundary
    - total cost function: C \* L2(w) + 1/N \* sum(hinge_loss)
- To deal with non-linearity: Kernel trick
    - transform original space into a different (higher dim) space may solve linear-separability problem
    - but trying all combinations of transformations is not possible
        - solution: use kernel functions to transform implicity (proof: duality)
        - Radial Basis Function kernel (such as Gaussian)
            - by varying hyperparameter sigma: choose between smooth/curvy decision boundary in original space

### kNN
- non-parametric, supervised
- classification: find k closest training examples, return majority label
- regression: average label of k closest examples
- distance function: euclidean distance, negative cosine similarity, etc.
- due to curse of dimensionality in high dimensions, try to do PCA first
    - Euclidean distance fails at high dim: there is little difference in the distances between different pairs of samples: all distances are similar in high dimensions):

### RBM (not in the book):

- Boltzmann Machine: resembles a simplified version MLP
- Features a visible input layer and a hidden layer
- Two-layer neural net that makes stochastic decisions as to whether a neuron should be on or off.
- Nodes are connected across layers, but no two nodes of the same layer are connected.