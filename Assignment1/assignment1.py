from Assignment1.functions import LoadBatch
from Assignment1.functions import softmax
import numpy as np

LEARNING_RATE = 0.1


class NeuralNet:

    def __init__(self, learning_rate, input_size, output_size, weights=0, bias=0):
        self.lr = learning_rate
        mu, sigma = 0, 0.01
        #self.bias = np.random.normal(mu, sigma, output_size)
        #self.weights = np.random.normal(mu, sigma, (output_size, input_size))
        if weights is 0:
            self.weights = np.random.normal(mu, sigma, (output_size, input_size+1))
        else:
            bias = np.array(bias).reshape(np.size(bias), 1)
            print(bias)
            self.weights = np.column_stack((weights, bias))

    def evaluate_classifier(self, X):
        return softmax(self.compute_input(X))

    def compute_input(self, X):
        X = np.concatenate((X, [np.ones(np.size(X, axis=1))]), axis=0)
        S = self.weights.dot(X)
        return S

    def compute_cost(self, X, Y, penalty_factor):
        norm_factor = 1/np.size(X, axis=1)
        S = NeuralNet.compute_input(self, X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(S, axis=1)

        print(np.size(S[:, 1]), np.size(Y[:, 1]))
        for i in range(np.size(Y, axis=1)):
            sum_entropy += NeuralNet.cross_entropy(self, S[:, i], Y[:, i])

        # TODO: Should bias be included???, not removed from the weights
        penalty_term = penalty_factor*np.sum(self.weights[..., :-1]*self.weights[..., :-1])

        return norm_factor*sum_entropy + penalty_term

    def cross_entropy(selfs, s, y):
        #s: network_output
        #y: expected output - one-hot encoding
        p = softmax(s)
        return -np.log10(np.transpose(y).dot(p))


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

def getDataSize(X, Y):
    output_size = np.size(X, axis=0)
    input_size = np.size(Y, axis=0)
    return input_size, output_size

def ComputeCost(X, Y, W, b, penalty_factor):
    input_size, output_size = getDataSize(X, Y)
    neural_net = NeuralNet(0, input_size, output_size, weights=W, bias=b)
    return neural_net.compute_cost(X, Y, penalty_factor)


if __name__ == '__main__':
    training_data = load_batch('data_batch_1')
    validation_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')

    processed_training_data = pre_process(training_data)
    processed_validation_data = pre_process(validation_data)
    processed_test_data = pre_process(test_data)

    #print(np.size(processed_training_data[0], axis=1), np.size(processed_training_data[1], axis=1))

    output_size = np.size(processed_training_data[1], axis=0)
    input_size = np.size(processed_training_data[0], axis=0)

    input_data = processed_training_data[0]

    neural_net = NeuralNet(LEARNING_RATE, input_size, output_size)
    penalty_factor = 0.001
    J = neural_net.compute_cost(processed_training_data[0], processed_training_data[1], penalty_factor)

    print(J)

    '''W = [[0, 0],
        [0, 0],
        [0, 0]]
    print(W)
    b = [1, 1, 1]
    b = np.array(b).reshape(np.size(b), 1)
    print(b)
    print(np.column_stack((W, b)))'''

