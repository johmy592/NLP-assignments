# L1X: Maximum Entropy Classification
from __future__ import division

import sys
import json
import math
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

    # print ("Encoding X...")
    for i, article in enumerate(articles):
        for word in set(article):
            if word in w2i:
                X[i, w2i[word]] = 1
    # print ("+ Finished encoding X.")


    # print ("One-hot encoding y")
    # Sort list of unique labels and map them to index
    label_map = dict([(label, i) for i, label in enumerate(sorted(list(set(labels))))])
    for i, label in enumerate(labels):
        y[i, label_map[label]] = 1
    # print ("+ Finished one-hot encoding y")

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


#-------------
# Core Layers
#-------------

class Dense(object):
    def __init__(self, n_units, input_shape=None, eta=0.001, rho=0.01, momentum=0.8):
        self.n_units = n_units          # Number of 'neurons' in the layer
        self.input_shape = input_shape  # Shape of layer input
        self.eta = eta                  # Learning rate
        self.rho = rho                  # Regularization parameter
        self.momentum = momentum        # Weight update momentum paramenter
        self.W = None                   # Layer weights
        self.w_updt = None              # Update step

    def init_weights(self):
        # Initialize weights between -1/sqrt(N) and 1/sqrt(N)
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w_updt = np.zeros_like(self.W)

    def forward_pass(self, X):
        self.layer_input = X
        return X.dot(self.W)

    def backward_pass(self, grad):
        # Calculate gradient with respect to layer inputs before adapting W
        grad_wrt_input = grad.dot(self.W.T)

        # Gradient w.r.t. layer's weights
        grad_w = self.layer_input.T.dot(grad)

        # Use momentum to calculate weight update
        self.w_updt = self.momentum * self.w_updt + (1 - self.momentum) * grad_w

        # Move against the gradient to minimize loss (use regularization)
        self.W -= self.eta * (self.w_updt + self.rho * self.W)

        return grad_wrt_input

    def output_shape(self):
        return (self.n_units, )

#-------------------
# Activation Layers
#-------------------

class Softmax(object):
    def softmax(self, X):
        E = np.exp(X - np.max(X, axis=1, keepdims=True))
        return E / E.sum(axis=1, keepdims=True)

    def forward_pass(self, X):
        self.layer_input = X
        return self.softmax(X)

    def backward_pass(self, grad):
        p = self.softmax(self.layer_input)
        return grad * p * (1 - p)

    def output_shape(self):
        return self.input_shape

class ReLU(object):
    def forward_pass(self, X):
        self.layer_inputs = X
        return np.where(X >= 0, X, 0)

    def backward_pass(self, grad):
        return grad * np.where(self.layer_inputs >= 0, 1, 0)

    def output_shape(self):
        return self.input_shape

#---------------
# Loss Function
#---------------

class CrossEntropyLoss(object):
    def loss(self, y, p):
        eps = 1e-16
        return - y * np.log(p + eps) - (1 - y) * np.log(1 - p + eps)

    def acc(self, y, p):
        return np.mean(np.argmax(y, axis=1) == np.argmax(p, axis=1))

    def gradient(self, y, p):
        eps = 1e-16
        return - (y / (p + eps)) + (1 - y) / (1 - p + eps)


#------------
# Base Model
#------------

class NeuralNetwork(object):
    def __init__(self):
        self.layers = []
        self.loss_function = CrossEntropyLoss()

    def predict(self, X):
        """Returns the most probable class for the given input."""
        y_pred = np.argmax(self.forward_pass(X), axis=1)
        return y_pred

    def add_layer(self, layer):
        """Adds a layer to the neural network"""

        # If layers exists set input shape to prev. layer's output shape
        if self.layers:
            layer.input_shape = self.layers[-1].output_shape()

        # Intialize layer's weights if they exist
        if hasattr(layer, 'W'):
            layer.init_weights()

        self.layers.append(layer)

    def backward_pass(self, grad):
        """
        Propogate the gradient backwards through the network and updates the weights
        """
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)

    def forward_pass(self, X):
        """Propogate the signal forward through the network"""
        out = X
        for layer in self.layers:
            out = layer.forward_pass(out)
        return out

    def train(self, X, Y, n_epochs=1, batch_size=1, eta=0.01, rho=0.1):
        """Trains a new classifier."""
        for epoch in range(n_epochs):
            loss, acc = [], []
            for X_batch, y_batch in minibatches(X, y, batch_size):
                prediction = self.forward_pass(X_batch)

                # Save loss and accuracy
                loss.append(np.mean(self.loss_function.loss(y_batch, prediction)))
                acc.append(self.loss_function.acc(y_batch, prediction))

                # Get the gradient of the loss w.r.t the prediction
                loss_grad = self.loss_function.gradient(y_batch, prediction)

                self.backward_pass(loss_grad)

            print ('[Epoch %d - Loss: %.2f, Acc: %.2f]' % (epoch, np.mean(loss), np.mean(acc)))


# Problem 3
if __name__ == "__main__":

    # Seed the random number generator to get reproducible results
    np.random.seed(42)

    # Load the training data and featurize it
    training_data = load_data(sys.argv[1])
    w2i = build_w2i(training_data)
    X, y = featurize(training_data, w2i)

    n_features = X.shape[1]
    n_outputs = y.shape[1]


    clf = NeuralNetwork()

    # Neural net architecture
    clf.add_layer(Dense(16, input_shape=(n_features, )))
    clf.add_layer(ReLU())
    clf.add_layer(Dense(16))
    clf.add_layer(ReLU())
    clf.add_layer(Dense(n_outputs))
    clf.add_layer(Softmax())

    print ("Training NN:")

    clf.train(X, y, 6, 32, 0.01, 0.1)

    # Compute the accuracy of the trained classifier on the test data
    test_data = load_data(sys.argv[2])
    X, y = featurize(test_data, w2i)
    print(np.mean(clf.predict(X) == np.argmax(y, axis=1)))
