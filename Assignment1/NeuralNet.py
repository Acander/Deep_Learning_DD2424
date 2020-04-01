from Assignment1.functions import softmax
from Assignment1.functions import LoadBatch
import numpy as np

LEARNING_RATE = 0.1


class NeuralNet:

    def __init__(self, learning_rate, input_size, output_size, weights=0, bias=0):
        self.lr = learning_rate
        mu, sigma = 0, 0.01
        # self.bias = np.random.normal(mu, sigma, output_size)
        # self.weights = np.random.normal(mu, sigma, (output_size, input_size))
        if weights is 0:
            self.weights = np.random.normal(mu, sigma, (output_size, input_size + 1))
        else:
            bias = np.array(bias).reshape(np.size(bias), 1)
            self.weights = np.column_stack((weights, bias))

    def evaluate_classifier(self, X):
        return softmax(self.compute_input(X))

    def compute_input(self, X):
        X = np.concatenate((X, [np.ones(np.size(X, axis=1))]), axis=0)
        S = self.weights.dot(X)
        return S

    def compute_cost(self, X, Y, penalty_factor):
        norm_factor = 1 / np.size(X, axis=1)
        S = NeuralNet.compute_input(self, X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(S, axis=1)

        # print(np.size(S[:, 1]), np.size(Y[:, 1]))
        for i in range(np.size(Y, axis=1)):
            sum_entropy += NeuralNet.cross_entropy(self, S[:, i], Y[:, i])

        # TODO: Should bias be included???, not removed from the weights
        penalty_term = penalty_factor * np.sum(self.weights[..., :-1] * self.weights[..., :-1])

        return norm_factor * sum_entropy + penalty_term

    def cross_entropy(self, s, y):
        # s: network_output
        # y: expected output - one-hot encoding
        P = softmax(s)
        return -np.log10(np.transpose(y).dot(P))

    def compute_accuracy(self, X, y):
        P = self.evaluate_classifier(X)
        correct_answers = 0
        assert np.size(P, axis=1) == np.size(X, axis=1)
        assert np.size(P, axis=1) == np.size(y)
        for i in range(np.size(P, axis=1)):
            highest_output_node = np.where(P[:, i] == np.amax(P[:, i]))
            if highest_output_node[0][0] == y[i]:
                correct_answers += 1

        return correct_answers / np.size(P, axis=1)

    def compute_gradients(self, X_batch, Y_batch, P_batch, penalty_factor):
        # P_batch = self.evaluate_classifier(X_batch)
        G_batch = Y_batch - P_batch
        gradient_W = 1 / np.size(X_batch) * G_batch.dot(X_batch.transpose())
        gradient_b = 1 / np.size(X_batch) * G_batch.dot(np.identity(np.size(X_batch, axis=1)))
        return [gradient_W + 2 * penalty_factor * np.delete(self.weights, np.size(self.weights, axis=1) - 1, 1),
                gradient_b]

    '''Remember here that the biases actually are actually the last column of the weight matrix'''

    def get_weights(self):
        return np.delete(self.weights, np.size(self.weights, axis=1) - 1, 1)

    def get_bias(self):
        return self.weights[:, np.size(self.weights, axis=1) - 1]


# n is the number of images
# d is the dimensionality of each image (3072=32x32x3)
def load_batch(filename):
    dict = LoadBatch(filename)
    X = np.array(dict[b'data'])  # matrix, dim: image pixel data, # of images * dim
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


# TODO: Check if that mean becomes approximatly zero is okay
# Standard Score? (X-mean)/std
def pre_process(training_data):
    [X, Y, y] = training_data
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)

    X = X - np.tile(mean_X, (np.size(X, axis=0), 1))
    X = X / np.tile(std_X, (np.size(X, axis=0), 1))
    return [X, Y, y]


def getDataSize(W):
    output_size = np.size(W, axis=0)
    input_size = np.size(W, axis=1)
    return input_size, output_size


def ComputeCost(X, Y, W, b, penalty_factor):
    input_size, output_size = getDataSize(W)
    neural_net = NeuralNet(0, input_size, output_size, weights=W, bias=b)
    return neural_net.compute_cost(X, Y, penalty_factor)


def ComputeAccuracy(X, y, W, b):
    input_size, output_size = getDataSize(W)
    neural_net = NeuralNet(0, input_size, output_size, weights=W, bias=b)
    return neural_net.compute_accuracy(X, y)


def ComputeGradients(X, Y, penalty_factor, batch_size):
    input_size = np.size(X, axis=0)
    output_size = np.size(Y, axis=0)
    neural_net = NeuralNet(0, input_size, output_size)
    batch_X = np.array(X[:, 0:0 + batch_size])
    batch_Y = np.array(Y[:, 0:0 + batch_size])
    P_batch = np.array(neural_net.evaluate_classifier(batch_X))
    grad_analytiaclly = neural_net.compute_gradients(batch_X, batch_Y, P_batch, penalty_factor)
    grad_numerically = ComputeGradsNumSlow(batch_X, batch_Y, P_batch, neural_net.get_weights(), neural_net.get_bias(),
                                           penalty_factor, np.exp(-6))
    return grad_analytiaclly, grad_numerically


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]


def printOutGradients(grad_analytically, grad_numerically):
    [grad_W, grad_b] = grad_analytically
    [grad_Wn, grad_bn] = grad_numerically
    print("Analytical (my own):.............")
    print(np.size(grad_W, axis=1))
    print(np.size(grad_b))
    print(grad_W)
    print(grad_b)
    print("Numerical (approximate):................")
    print(np.size(grad_Wn, axis=1))
    print(np.size(grad_bn))
    print(grad_Wn)
    print(grad_bn)


if __name__ == '__main__':
    training_data = load_batch('data_batch_1')
    validation_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')

    processed_training_data = pre_process(training_data)
    processed_validation_data = pre_process(validation_data)
    processed_test_data = pre_process(test_data)

    # print(np.size(processed_training_data[0], axis=1), np.size(processed_training_data[1], axis=1))

    output_size = np.size(processed_training_data[1], axis=0)
    input_size = np.size(processed_training_data[0], axis=0)

    input_data = processed_training_data[0]

    neural_net = NeuralNet(LEARNING_RATE, input_size, output_size)
    penalty_factor = 0.001
    batch_size = 1
    grad_analytically, grad_numerically = ComputeGradients(processed_training_data[0], processed_training_data[1],
                                                           penalty_factor, batch_size)
    printOutGradients(grad_analytically, grad_numerically)

    # J = neural_net.compute_cost(processed_training_data[0], processed_training_data[1], penalty_factor)
    '''W = [[0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]]
    print(W)
    #b = [1, 1, 1]
    #b = np.array(b).reshape(np.size(b), 1)
    print(np.delete(W, np.size(W, axis=1)-1, 1))
    print(W)
    #print(b)
    #print(np.column_stack((W, b)))'''
