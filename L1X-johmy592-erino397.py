# L1X: Maximum Entropy Classification
from __future__ import division

import json
import numpy as np

def load_data(filename):
    """Loads movie review data."""
    result = []
    with open(filename) as source:
        for review in json.load(source):
            text = review['text'].split()
            polarity = review['polarity']
            result.append((text, polarity))
    return result

# Problem 1

# In order to represent a movie review as a NumPy vector, we need to
# assign to each word in our training set a unique integer index which
# will give us that component in the vector which is 1 if the
# corresponding word is present in the review, and 0 otherwise.

def build_w2i(data):
    """Returns a dictionary that maps words to integers."""
    # TODO: Replace the following line with your own code

    vocab =  set([word for art, _ in data for word in set(art)])
    # Get unique words and sort them alphabetically
    vocab = sorted(list(set(vocab)))
    idx_map = dict([(word, i) for i, word in enumerate(vocab)])
    return idx_map

def featurize(data, w2i):
    """Converts review data into matrix format

    The argument w2i is a word index, as described above. The function
    should return a pair of NumPy matrices X, Y, where X is an N-by-F
    matrix (N: number of instances in the data, F: number of features,
    here: number of unique words in the data), and where Y is an
    N-by-K matrix (K: number of classes, here: 2).

    """
    # TODO: Replace the following line with your own code
    articles = [art for art, _ in data]
    labels = [label for _, label in data]

    X = np.zeros((len(articles), len(w2i)))
    y = np.zeros((len(articles), len(set(labels))))

    print ("Encoding X...")
    for i, article in enumerate(articles):
    	for word in set(article):
    		if word in w2i:
    			X[i, w2i[word]] = 1
    print ("+ Finished encoding X.")


    print ("One-hot encoding y")
    # Sort list of unique labels and map them to index
    label_map = dict([(label, i) for i, label in enumerate(sorted(list(set(labels))))])
    for i, label in enumerate(labels):
    	y[i, label_map[label]] = 1
    print ("+ Finished one-hot encoding y")

    return X, y

# Problem 2

def minibatches(X, Y, batch_size):
    """Yields mini-batches from the specified X and Y matrices."""
    m = X.shape[0]
    n_batches = int(np.floor(m / batch_size))
    random_indices = np.random.permutation(np.arange(m))
    for i in range(n_batches):
        batch_indices = np.arange(i * batch_size, (i + 1) * batch_size)
        batch_indices = random_indices[batch_indices]
        yield X[batch_indices], Y[batch_indices]

def softmax(X):
    """Computes the softmax function for the specified batch of data.

    The implementation uses a common trick to improve numerical
    stability; this trick is explained here:

    http://stackoverflow.com/a/39558290

    """
    E = np.exp(X - np.max(X, axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

class MaxentClassifier(object):

    def __init__(self, n_features, n_classes):
        self.W = np.zeros((n_features, n_classes))

    def p(self, X):
        """Returns the class probabilities for the given input."""
        class_probs = softmax(X.dot(self.W))
        return class_probs

    def predict(self, X):
        """Returns the most probable class for the given input."""
        y_pred = np.argmax(self.p(X), axis=1)
        return y_pred

    def train(self, X, Y, n_epochs=1, batch_size=1, eta=0.01, rho=0.1):
        """Trains a new classifier."""
        for epoch in range(n_epochs):
        	for X_batch, y_batch in minibatches(X, y, batch_size):

        		# Display training accuracy
        		#print (np.sum(self.predict(X) == np.argmax(y, axis=1))/len(X))

        		# Gradient of cross entropy loss w.r.t. layer weights W
        		grad_W = - X_batch.T.dot(y_batch - softmax(X_batch.dot(self.W)))
        		self.W -= eta * (grad_W + rho * self.W)


# Problem 3
if __name__ == "__main__":
    import sys

    # Seed the random number generator to get reproducible results
    np.random.seed(42)

    # Load the training data and featurize it
    training_data = load_data(sys.argv[1])
    w2i = build_w2i(training_data)
    X, y = featurize(training_data, w2i)

    # Train the classifier
    classifier = MaxentClassifier(X.shape[1], y.shape[1])

    classifier.train(X, y, 5, 18, 0.01, 0.1)

    # Compute the accuracy of the trained classifier on the test data
    test_data = load_data(sys.argv[2])
    X, y = featurize(test_data, w2i)
    print(np.mean(classifier.predict(X) == np.argmax(y, axis=1)))
