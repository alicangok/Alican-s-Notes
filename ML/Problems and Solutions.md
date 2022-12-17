### Kernel regression

- with 1-2 D input/output data, we can try polynomial regression, and it's easy to see whether function fits the data
- but with multiple dimensional features, finding the right polynomial is hard
- kernel regression: non-parametric, based on data (like kNN)
- f(x) = sum(kernels centered around data points), such as gaussian kernel
    - tune a variance hyperparameter (that determines spread of gaussian)

### Multiclass classification

- logistic regression -> softmax
- kNN
- SVM (normally binary classifier): one versus rest approach
    - for n classes, you train n classifiers
    - pick the most certain prediction (max distance = (wx+b) / ||w||)
- Decision Trees
    - Hierarchical classification trees
- Naive Bayes

### One-class classification

- outlier detection, anomaly detection
- example: normal traffic in a network, tons of examples
    - but very litle examples of attacks
- one-class Gaussian, kmeans, kNN, SVM
- one-class Gaussian: max likelihood estimation of a multivariate gaussian distr
    - if prob(sample) < threshold: classify as outlied
    - mixture of gaussians also possible
- one-class kmeans: build model, and define threshold of distance to center
- one-class kNN: e.g. is distance(z, NN(z)) > C x distance(NN(z), NN(NN(z))) ?
- one-class SVM: two methods
    - separate all examples from the origin in feature space
    - obtain a spherical boundary around data by minimizing volume

### Multi-label classification

- e.g. multiple labels for a picture
- threshold: a hyperparameter to be learned
- use binary cross-entropy as cost
- output layer of many units: one unit per label
- alternatively, if only a few possible labels exist, treat each combination as a class (and transform the problem into a multiclass problem)
    - this way, the labels are correlated

### Ensemble learning

- instead of one super-accurate model, train many low-accuracy models
- combine them to obtain a high-accuracy meta-model
- two widely used methods
    - random forest: 3 steps: bagging / training each / majority voting
        - bagging: combine models built on subsets by bootstrapping
            - many slightly different versions of training data
            - bootstrap: resample with replacement to create subsets
            - need to tune number of trees and size of random subset
        - reduces variance, prevents overfitting
        - reduces effects due to over/underrepresentation, outliers
    - gradient boosting: sequential process, each subsequent model attempts to correct errors from previous model. This is done by giving higher weights to the observations which were incorrectly predicted. Final model (strong learner) is the weighted mean of all the models (weak learners). 
        - instead of getting gradients directly, we use residuals as proxy
            - residuals show us how the model has to be adjusted
        - boosting reduces bias, can overfit
            - gotta tune depth and number of trees
        - Gradient boosting very powerful, can handle huge datasets
        - usually outperforms random forest in accuracy, but slower to train
        - regression:
            - start with constant model f0, taking mean of all labels
            - then modify labels of each example by subtracting mean (model output)
                - this is called the residual -> the new label
            - build new decision tree model f1:  f = f0 + α f1
                - α is the learning rate (a hyperparameter)
            - recompute residuals, build new decision tree model f2
                - overall model f = f0 + α f1 + α f2
            - continue until M (another hyperparameter) trees are combined
        - classification:
            - initial constant model (f = p/(1-p) from labels)
            - at each iteration, add a new tree
                - replace labels by partial derivative of likelihood wrt current model
                - build new tree
                - update by choosing optimal update step to maximize likelihood

### Learning to label sequences

- annotate a sequence (individual words, whole sequence)
- RNN not the only possible model
- CRF: conditional random fields
    - features typically constructed by hand
    -  slower to train, not applicable to large sets, outperformed by RNNs
    
### Sequence-to-sequence learning

- machine translation models, text summarization
- encoder / decoder architecture
    - encoder can be CNN or RNN -> generates embedding (representation)
    - decoder typically RNN
- attention: combine information from encoder & current state of decoder
    - better retention of long-term dependencies (than gated units only)

### Active Learning

- usually used when obtaining labels is costly (e.g. expert opinions required)
- initally start with few labeled examples
- then add labels only to those examples that contribute to model quality
- multiple strategies possible
    - 1) data density & uncertainty based
        - determine importance of each unlabeled example
            - importance(x) = density(x) * uncertainty (x)
            - uncertainty for binary, when prediction score is around 0.5
            - uncertainty for multiclass: entropy
        - ask expert to annotate the most important ones
    - 2) support vector based
        - build SVM model on labeled data
        - ask expert to annotate the hardest examples (closest to hyperplane)

### Semi-supervised learning

- a fraction of the dataset is labeled
    - goal: use unlabeled examples to improve model performance
- Self learning (limited success)
    - build initial model
    - add unlabeled examples with high confidence predictions to training set
- Autoencoders: goal is to regenerate input
    - bottleneck layer: refined embedding
- Denoising autoencoder
    - corrupt training example (noise), try to regenerate
- Ladder network: upgraded denoising autoencoder
    - encoder and decoder same number of layers
    - Bottleneck layer: directly predict label (softmax)
    - cost function: penalizes diff between enc/dec output differences

### One-shot learning

- typically used for face recognition
    - two photos: same person?
- siamese neural network
    - triplet loss: max( || f(A) - f(P) || - || f(A) - f(N) || + α, 0)
- training:
    - strategy: choose tougher negative examples each epoch -> quicker training
- One-shot learning actually requires more than one example per entity to train
    - but in deployment, require one example

### Zero-shot learning

- OOV problem
- Trick: use embeddings not just for input, but also for labels
- Could work for image labelling, when the word encoder is trained with word2vec
    - word embeddings contain semantic data, which could match with image features