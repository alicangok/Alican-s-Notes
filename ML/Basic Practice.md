### Feature engineering
- raw data -> dataset
- requires creativity, domain knowlege
- one hot encoding (binary label for each feature)
- normalization to (0,1): solve numerical problems, prevents a feature from dominating update
- standardization: to normal distribution
- rules of thumb
    - for unsupervised problems, standardization is usually better
    - when there's outliers, standardization better (simple normalization squeezes everything)
- missing features: either remove examples / learn missing features / use average values 

### Learning algorithm
Certain considerations:
- explainability important?
- train all at once? incremental learning? memory considerations
- data linearly separable, or linear model possible? simple SVM/regression will do
- prediction speed?

### Sets
- training/validation/test
- training: to learn parameters
- validation: to choose the learning algorithm and find best hyperparameters
- test: final assessment before delivery to client / production

### Under/overfitting
- low bias: predicts training data well
- high bias: many mistakes on training data, underfit. Possible reasons:
    - model too simple
    - features not informative enough
- overfitting: predicts training data well, but poor for validation/test
    - model too complex
    - too many features for a small number of examples
    - also called high variance (from statistics)
        - sensitivity to small fluctuations in the training set
        - if training data vas sampled differently, the learned model would be significantly different
- solutions for overfitting:
    - regularization
    - simpler model
    - reduce dimensionality of examples
    - add more training data

### Regularization
- goal: force the learing algorithm to build a less complex model:
    - slightly increase bias, but significantly reduce variance
    - "bias-variance tradeoff"
- L1 regularization (Lasso): add sum(abs(w)) cost to objective function
    - produces a sparse model (most parameters)
    - kind of a feature selection (increases model explainability)
- L2 regularization (Ridge): add sum(w-squared) to objective function
    - better for performance on validation/test sets
    - differentiable
- dropout
- batch normalization
- data augmentation
- early stopping

### Performance assessment
- is our model generalizing well?
- regression: MSE
- classification: confusion matrix, accuracy, precision/recall, area under ROC curve
- [[F1 score, precision and recall]]
- ROC: sweep threshold, measure area under true positive rate (recall) and false positive rate (FP/(FP+TN))
    - no skill: y=x line (AUC = 0.5)
    - we should be well above it

### Hyperparameter Tuning
- grid search / random search / Bayesian hyperparameter optimization / others
- cross-validation:
    - step 1: fix hyperparameters
    - step 2: split training set into subsets (folds), e.g. five-fold (hold one out)
    - step 3: average the 5 validation set metrics
    - step 4: repeat 1-3 till you find a good hyperparameter combination
    - step 5: use entire training set to build the final model
