Challenge: No labels -> no solid reference point to judge quality

### Density estimation

- parametric (gaussian fitting)
- nonparametric (like in [[Problems and Solutions#Kernel regression|kernel regression]])
    - use leave one out estimate / stats to find kernel hyperparameter

### Clustering

Problem: Assign labels using an unlabeled dataset
- k-means: centroid based
    - randomly pick k centroids
    - assign each example to closest centroid
    - update centroid (by averaging example features)
    - repeat..
    - finding optimal k: various techniques, educated guesses 
- DBSCAN: density based clustering with noise
- hierarchical clustering
- determining number of clusters, multiple methods:
    - Prediction strength:
        - split data into two, run clustering algorithm twice
        - choose number of classes with best prediction strength
        - ~ worst class performance of binary co-membership matrix
            - matrix of size  `n_test` x `n_test`, indicates whether two samples fall into the same cluster on the test set
            - for the "one"s in the matrix, see if they also share clusters in training
        - pick largest k such that pred strength >0.8
    - Elbow method:
        - Calculate the Within-Cluster-Sum of squared errors for different k
        - choose k for which WSS improvement starts to diminish: like elbow in plot
- GMM: each example has a membership score with several clusters
    - expectation maximization : optimize maximum likelihood criterion
        - 4 steps: likelihood / bayes / update mean vars / mixture probs
        - expectation: creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters (calculate likelihoods of each observation and use Bayes rule to assign each observation to a cluster)
        - maximization: computes parameters maximizing the expected log-likelihood found on the E step (update gaussian vars & mixture probs)

### Dimensionality reduction

Reducing dimensionality is more useful for visualization and interpretability nowadays, now that computation is usually no longer a problem.
- PCA: principle component vectors (new coordinate system)
    - eigenvectors of the covariance matrix (cov = E[ (x-mean(x)) * (y-mean(y)) ] )
        - correlation matrix requires dividing by standard deviations as well
- t-SNE & UMAP: requires similarity metric (not simply euclidean)
    - UMAP compresses the 2D graph
    - [[PCA vs t-SNE]]
- LDA, CCA
- autoencoders / bottleneck layer

### Anomaly detection

- train an autoencoder on typical set
    - if reconstruction error of test example too high -> outlier
- one-class classification: [[Problems and Solutions#One-class classification]]

(do additional search here for literature)