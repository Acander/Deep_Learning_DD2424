from Assignment1.functions import softmax
from Assignment1.functions import LoadBatch
import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:

    def __init__(self, input_size, output_size, weights=0, bias=0):
        mu, sigma = 0, 0.01
        # self.bias = np.random.normal(mu, sigma, output_size)
        # self.weights = np.random.normal(mu, sigma, (output_size, input_size))
        if weights is 0:
            self.weights = np.random.normal(mu, sigma, (output_size, input_size + 1))
        else:
            #print("I ran!")
            bias = np.array(bias).reshape(np.size(bias), 1)
            self.weights = np.column_stack((weights, bias))

    def evaluate_classifier(self, X):
        return softmax(self.compute_input(X))

    def compute_input(self, X):
        b = np.matrix(self.get_bias()).transpose()
        sum_matrix = np.matrix(np.ones(np.size(X, axis=1)))
        S = self.get_weights().dot(X) + b.dot(sum_matrix)
        return S

    def compute_cost(self, X, Y, penalty_factor):
        norm_factor = 1 / np.size(X, axis=1)
        S = NeuralNet.compute_input(self, X)
        #print(S)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(S, axis=1)

        # print(np.size(S[:, 1]), np.size(Y[:, 1]))
        for i in range(np.size(Y, axis=1)):
            sum_entropy += NeuralNet.cross_entropy(self, S[:, i], Y[:, i])

        penalty_term = penalty_factor * np.sum(np.square(self.get_weights()))
        return norm_factor * sum_entropy + penalty_term

    def cross_entropy(self, s, y):
        # s: network output
        # y: expected output - one-hot encoding
        p = np.array(softmax(s))
        return -np.log10(y.dot(p)[0])

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

    def compute_gradients(self, X_batch, Y_batch, P_batch, penalty_factor, batch_size):
        G_batch = np.array(-(Y_batch - P_batch))
        gradient_W = G_batch.dot(X_batch.transpose())/batch_size
        gradient_b = np.sum(G_batch, axis=1)/batch_size
        return [gradient_W + 2 * penalty_factor * self.get_weights(),
                gradient_b]

    '''Remember here that the biases actually are actually the last column of the weight matrix'''

    def get_weights(self):
        return np.delete(self.weights, np.size(self.weights, axis=1) - 1, 1)

    def get_bias(self):
        return np.array(self.weights[:, np.size(self.weights, axis=1) - 1])

    def MiniBatchGD(self, X, Y, X_val, Y_val, penalty_factor, GDparams):
        batch_size, eta, n_epochs = GDparams
        train_loss = np.zeros(n_epochs)
        validation_loss = np.zeros(n_epochs)
        for i in range(n_epochs):
            self.fit(X, Y, penalty_factor, eta, batch_size)
            ts = self.compute_cost(X, Y, penalty_factor)
            vl = self.compute_cost(X_val, Y_val, penalty_factor)
            print("Epoch: ", i)
            print("Train_loss: ", ts)
            print("Validation_loss: ", vl)
            train_loss[i] = ts
            validation_loss[i] = vl

        return train_loss, validation_loss


    def fit(self, X, Y, penalty_factor, eta, batchSize=-1):

        if (batchSize == -1):
            batchSize = X.shape[1]
        for i in range(0, X.shape[1], batchSize):
            batchX = X[:, i:i + batchSize]
            batchY = Y[:, i:i + batchSize]
            batchP = self.evaluate_classifier(batchX)
            [grad_W, grad_b] = self.compute_gradients(batchX, batchY, batchP, penalty_factor, batchSize)
            weights = self.get_weights() - eta*grad_W
            bias = self.get_bias() - eta*grad_b
            self.weights = np.column_stack((weights, bias))


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
    std_X = np.std(X, ddof=1, axis=0)

    X = X - np.tile(mean_X, (np.size(X, axis=0), 1))
    X = X / np.tile(std_X, (np.size(X, axis=0), 1))

    return [X, Y, y]


def getDataSize(W):
    output_size = np.size(W, axis=0)
    input_size = np.size(W, axis=1)
    return input_size, output_size


def ComputeCost(X, Y, W, b, penalty_factor):
    input_size, output_size = getDataSize(W)
    neural_net = NeuralNet(input_size, output_size, weights=W, bias=b)
    return neural_net.compute_cost(X, Y, penalty_factor)


def ComputeAccuracy(X, y, W, b):
    input_size, output_size = getDataSize(W)
    neural_net = NeuralNet(input_size, output_size, weights=W, bias=b)
    return neural_net.compute_accuracy(X, y)


def ComputeGradients(X, Y, penalty_factor, batch_size):
    input_size = np.size(X, axis=0)
    output_size = np.size(Y, axis=0)
    neural_net = NeuralNet(input_size, output_size)
    batch_X = np.array(X[:, 0:0 + batch_size])
    batch_Y = np.array(Y[:, 0:0 + batch_size])
    P_batch = np.array(neural_net.evaluate_classifier(batch_X))
    #print(P_batch)
    grad_analytiaclly = neural_net.compute_gradients(batch_X, batch_Y, P_batch, penalty_factor, batch_size)
    grad_numerically = ComputeGradsNumSlow(batch_X, batch_Y, P_batch, neural_net.get_weights(), neural_net.get_bias(),
                                           penalty_factor, 0.000001)
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

def gradient_difference(gradient_analytical, gradient_numeric, eps):
    numerator = np.absolute(gradient_analytical - gradient_numeric)
    denominator = np.max(np.array([eps, (np.absolute(gradient_analytical) + np.absolute(gradient_numeric))]))
    print("Numerator: ", numerator)
    print("Denominator: ", denominator)
    print("eps: ", eps)
    print("abs: ", (np.absolute(gradient_analytical) + np.absolute(gradient_numeric)))
    return numerator/denominator


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

def print_gradient_check(grad_W, grad_Wn, grad_b, grad_bn, eps):
    print("Gradient differences:")
    print("Weights:")
    for i in range(np.size(grad_W, axis=0)):
        for j in range(np.size(grad_W, axis=1)):
            print(gradient_difference(grad_W[i][j], grad_Wn[i][j], eps))

    print("Bias:")
    for i in range(np.size(grad_b, axis=0)):
        print(gradient_difference(grad_b[i], grad_bn[i], eps))

def plot_loss(train_loss, val_loss):
    plt.plot(np.arange(np.size(train_loss)), train_loss, color='blue', label='Training Loss')
    plt.plot(np.arange(np.size(val_loss)), val_loss, color='red', label='Validation Loss')

    xMin = 0
    xMax = train_loss.size

    yMin = np.min(np.concatenate((train_loss, val_loss)))
    yMax = np.max(np.concatenate((train_loss, val_loss)))

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)

    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

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
    neural_net = NeuralNet(input_size, output_size)

    tdi = processed_training_data[0]
    tdl = processed_training_data[1]
    vdi = processed_validation_data[0]
    vdl = processed_validation_data[1]

    batch_size = 10
    eta = 0.1
    n_epochs = 100
    GDparams = batch_size, eta, n_epochs
    penalty_factor = 0.01

    train_loss, validation_loss = neural_net.MiniBatchGD(tdi, tdl, vdi, vdl, penalty_factor, GDparams)
    plot_loss(train_loss, validation_loss)

    print("-----------------------------")
    print("Final train loss: ", neural_net.compute_cost(processed_training_data[0], processed_training_data[1], penalty_factor))
    print("Final validation loss: ", neural_net.compute_cost(processed_validation_data[0], processed_validation_data[1], penalty_factor))
    print("Final test loss: ", neural_net.compute_cost(processed_test_data[0], processed_test_data[1], penalty_factor))

    print("------------------------------")
    print("Final train accuracy: ", neural_net.compute_accuracy(processed_training_data[0], processed_training_data[2]))
    print("Final validation accuracy: ", neural_net.compute_accuracy(processed_validation_data[0], processed_validation_data[2]))
    print("Final test accuracy: ", neural_net.compute_accuracy(processed_test_data[0], processed_test_data[2]))

    '''
    input_data = processed_training_data[0]
    neural_net = NeuralNet(input_size, output_size)
    penalty_factor = 0.1
    batch_size = 50
    grad_analytically, grad_numerically = ComputeGradients(processed_training_data[0], processed_training_data[1],
                                                           penalty_factor, batch_size)
    printOutGradients(grad_analytically, grad_numerically)

    eps = 0.001
    [grad_W, grad_b] = grad_analytically
    [grad_Wn, grad_bn] = grad_numerically'''

    #print_gradient_check(grad_W, grad_Wn, grad_b, grad_bn, eps)

    #J = neural_net.compute_cost(processed_training_data[0], processed_training_data[1], penalty_factor)

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
