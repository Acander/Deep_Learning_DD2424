from Assignment1.functions import LoadBatch
from Assignment1.functions import softmax
import numpy as np

LEARNING_RATE = 0.1


class NeuralNet:

    def __init__(self, learning_rate, input_size, output_size):
        self.lr = learning_rate
        mu, sigma = 0, 0.01
        self.weights = np.random.normal(mu, sigma, (output_size, input_size + 1))
        #self.bias = np.random.normal(mu, sigma, output_size)

    def evaluate_classifier(self, X):
        return softmax(self.compute_input(X))

    def compute_input(self, X):
        #print(np.shape(X))
        #print(np.shape(self.weights))
        #print(np.shape(self.bias))
        #print(np.ones(np.size(X, axis=1)))
        X = np.concatenate((X, [np.ones(np.size(X, axis=1))]), axis=0)
        print(X)
        print(self.weights)
        print(self.weights.dot(X))
        s = self.weights.dot(X)
        print(np.shape(s))
        return s

# n is the number of images
# d is the dimensionality of each image (3072=32x32x3)
def load_batch(filename):
    dict = LoadBatch(filename)
    X = np.array(dict[b'data']) # matrix, dim: image pixel data, # of images * dim
    # Y = matrix, diM: # of images * # of labels, one-hot representation
    y = np.array(dict[b'labels'])  # vector, contains labels, dim: # of images

    Y = make_one_hot_encoding(len(X), y)

    return [np.array(X.astype(float)).transpose(), np.array(Y.astype(float)).transpose(), y.astype(float)]


def make_one_hot_encoding(batch_size, indices):
    Y = np.zeros((batch_size, 10))
    for i in range(batch_size):
        hot = indices[i]
        Y[i][hot] = 1

    return Y

#TODO: Check if that mean becomes approximatly zero is okay
#Standard Score? (X-mean)/std
def pre_process(training_data):
    [X, Y, y] = training_data
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)

    X = X - np.tile(mean_X, (np.size(X, axis=0), 1))
    X = X / np.tile(std_X, (np.size(X, axis=0), 1))
    return [X, Y, y]

if __name__ == '__main__':
    training_data = load_batch('data_batch_1')
    validation_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')

    processed_training_data = pre_process(training_data)
    processed_validation_data = pre_process(validation_data)
    processed_test_data = pre_process(test_data)

    output_size = np.size(processed_training_data[1], axis=0)
    input_size = np.size(processed_training_data[0], axis=0)

    input_data = processed_training_data[0]

    neural_net = NeuralNet(LEARNING_RATE, input_size, output_size)
    p = neural_net.evaluate_classifier(input_data)
    print(p)
