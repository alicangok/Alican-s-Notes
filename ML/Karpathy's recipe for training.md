From: [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)

Main principle: Prevent the introduction of a lot of “unverified” complexity at once

1) Inspect the data thoroughly
    - look for patterns
    - duplicate examples? imbalances? biases?
    - are labels correct? noisy?
    - preprocessing needed? downsampling?
    - write some simple code to search/filter/sort the data, visualize distributions, identify outliers
2) Set up end-to-end training/evaluation skeleton, get dumb (e.g. linear classifier, tiny convnet) baselines, visualize losses, metrics, predictions
    - fix seed (sanity)
    - run eval on entire test set (we are pursuing correctness and staying sane)
    - verify loss at initalization
    - init well, especially final layer (e.g. correct bias for imbalanced sets)
        - otherwise the network just learns the bias during the first few epochs
    - sanity check: set inputs to 0, does the model perform worse?
    - over fit a single batch of few examples, add layers/filters, try to get zero loss
        - at this point, do we get perfect labeling? otherwise, there's a bug
    - increase the capacity (layer/filter) a bit, does training loss go down?
    - visualize the data right before `y_hat = model(x)`, to make sure data is good
    - visualize model predictions/dynamics on a fixed test batch during training
        - does the network "wiggle" too much, revealing instabilities? reduce LR?
    - use backprop to chart dependencies
        - there's usually a lot of vectorized/broadcasted operations
        - make sure `view/transpose` is used correctly (otherwise batch dimension will get mixed in)
        - solution:
            - set the loss to the sum of all outputs of example i
            - run backward pass
            - the only non-zero gradient should be on the i'th input
            - for AR models, model at time t should only depend on 1...t-1
3) Overfit: first get a large model to overfit, get low training loss
    - pick an approproate architecture, don't be a hero early on
        - start with ResNet-50 for example
    - Adam with 3e-4 LR is safe
    - complexify one at a time
    - different problems require different LR decay strategies
        - initially, disable LR decays entirely, tune it all the way at the end
    - to gain confidence: visualize the first layer weights (shouldn't look like noise)
4) Regularize: Tradeoff some training loss for val loss
    - get more data if possible
    - try ensembles (some improvement up to 5 models)
    - data augmentation / creative augmentation
    - pretrain: almost never hurts, even when you have a lot of data
    - reduce input dimensionality, for low-level details may induce overfit for small datasets
    - smaller model size: FC layers at the top now replaced with avg pooling for images
    - decrease batch size if using batchnorm: ~ stronger regularization (more error in stats, wiggles)
    - dropout (dropout2d for CNNs): [be careful](https://arxiv.org/abs/1801.05134) if using batchnorm as well (var. shift)
    - weight decay (L1, L2): increase the penalty 
    - early stopping, based on measured validation loss
    - try a larger model: will of course overfit eventually, but early stopped versions may be better than smaller models
5) Tune: explore a wide model space for architectures
    - multiple hyperparameters: [random](http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf) over grid search
    - hyperparameter optimization: try Bayesian toolboxes
6) Squeeze out the juice, final tricks:
    - Ensembles, [distilling](https://arxiv.org/abs/1503.02531).
    - Leave it training, even after validation loss levels off 
