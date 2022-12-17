### Metric Learning

- choosing Euclidean dist / cosine similarity is actually arbitrary, just like squared error in linear regression
- tailored for problem / dataset
- positive semidefiniteness: required for metrics
    - this guarantees nonnegativity of distance and triangle inquality

### Learning to rank

- 3 approaches: pointwise, pairwise, listwise
    - pointwise: one example per document, supervised (not good)
    - pairwise: better, but far from perfect
    - listwise: directly optimize a metric that reflects quality of ranking, such as MAP

- mean average precision (over all classes)
    - used for object detection (localisation and classification)
        - precision: how many positive predictions are correct
        - IoU: intersection over union (area of overlap / union)
            - with different IoU thresholds, a TP can be FP
        - recall: how many positives can you catch
    - AP general definition: area under PR curve at a defined IoU
        - some competitions have multiple AP definitions, at different IoUs
### Learning to recommend

- factorization machines
- denoising autoencoders

### Self-supervised word embeddings

- word2vec
    - continuous bag of words: try to predict middle word
    - skip-gram: try to predict context from middle word
        - cost function: cross entropy (minimizes negative log likelihood)