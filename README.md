# Glove-
Glove For Spark 
Pennington et al. proposes a novel word representation algorithm called GloVe (Global Vectors
for Word Representation) that synthesizes the two primary model families for learning
vectors, matrix factorization methods over term-document matrices such as LSA (Deerwester
et al., 1990) and context-window modeling methods such as Word2Vec (Mikolov et al., 2014).
The goal of GloVe is to embed representations of words in a corpus into a continuous vector
space in such a manner that the parallel semantic relationships between words are modelled
by vector offsets between words. In other words, we desire that the produced vector space
encodes equivalent semantic relationships via linear offsets.
In this work I seek to distribute the training portion of the GloVe algorithm (i.e. developing the log-bi linear model given a co-occurrence matrix) using the Spark framework.
