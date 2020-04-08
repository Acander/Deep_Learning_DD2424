from Assignment1.functions import softmax
from Assignment1.functions import LoadBatch
from Assignment1.functions import montage
import numpy as np
import matplotlib.pyplot as plt

class ANN_two_layer:

    def __init__(self, input_size, hidden_size, output_size, weights_1=0, bias_1=0, weights_2=0, bias_2=0):
        mu, sigma = 0, 0.01
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if weights_1 is 0:
            weights_1 = np.random.normal(mu, sigma, (hidden_size, input_size))
            weights_2 = np.random.normal(mu, sigma, (output_size, hidden_size))
            bias_1 = np.zeros((hidden_size, 1))
            bias_2 = np.zeros((output_size, 1))
            self.weights_1 = np.column_stack(weights_1, bias_1)
            self.weights_1 = np.column_stack(weights_2, bias_2)
        else:
            # print("I ran!")
            bias_1 = np.array(bias_1).reshape(np.size(bias_1), 1)
            self.weights_1 = np.column_stack((weights_1, bias_1))
            bias_2 = np.array(bias_2).reshape(np.size(bias_2), 1)
            self.weights_2 = np.column_stack((weights_2, bias_2))

        self.hidden_layer_batch = np.matrix((hidden_size, 1))

    def evaluate_classifier(self, X):
        hidden_layer = max(self.compute_hidden(X), np.zeros((self.hidden_size, np.size(X, axis=1))))
        self.hidden_layer_batch = np.column_stack((self.hidden_layer_batch, hidden_layer))
        P = softmax(self.compute_output(hidden_layer))
        return P

    def compute_hidden(self, X):
        b1, b2 = self.get_bias()
        b1 = np.matrix(b1).transpose()
        sum_matrix = np.matrix(np.ones(np.size(X, axis=1)))
        w1, w2 = self.get_weights()
        S_1 = w1.dot(X) + b1.dot(sum_matrix)
        return S_1

    def compute_output(self, S_1):
        b1, b2 = self.get_bias()
        b1 = np.matrix(b1).transpose()
        sum_matrix = np.matrix(np.ones(np.size(S_1, axis=1)))
        w1, w2 = self.get_weights()
        S = w1.dot(S_1) + b1.dot(sum_matrix)
        return S

    def compute_cost(self, X, Y, penalty_factor):
        norm_factor = 1 / np.size(X, axis=1)
        P = self.evaluate_classifier(X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(P, axis=1)

        for i in range(np.size(Y, axis=1)):
            sum_entropy += self.cross_entropy(P[:, i], Y[:, i])

        w1, w2 = self.get_weights()
        penalty_term = penalty_factor * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return norm_factor * sum_entropy + penalty_term

    def cross_entropy(self, p, y):
        # s: softmax network output
        # y: expected output - one-hot encoding
        return -np.log10(y.dot(p)[0])

    def compute_total_loss(self, X, Y):
        P = self.evaluate_classifier(X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(P, axis=1)

        for i in range(np.size(Y, axis=1)):
            sum_entropy += self.cross_entropy(P[:, i], Y[:, i])

        return sum_entropy

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
        # We backprop the gradient G through the net #
        w1, w2 = self.get_weights()
        G_batch = self.init_G_batch(Y_batch, P_batch)

        dloss_W2, dloss_b2 = self.get_weight_gradient(self.hidden_layer_batch, G_batch, batch_size)
        G_batch = self.propagate_G_batch(G_batch, w1, self.hidden_layer_batch)
        dloss_W1, dloss_b1 = self.get_weight_gradient(X_batch, G_batch, batch_size)

        gradient_W1 = dloss_W1 + 2 * penalty_factor * w1
        gradient_b1 = dloss_b1

        gradient_W2 = dloss_W2 + 2 * penalty_factor * w2
        gradient_b2 = dloss_b2

        return [(gradient_W1, gradient_b1), (gradient_W2, gradient_b2)]

    def init_G_batch(self, Y_batch, P_batch):
        return np.array(-(Y_batch - P_batch))

    def get_weight_gradient(self, layer_input_batch, G_batch, batch_size):
        dloss_W = G_batch.dot(layer_input_batch.transpose()) / batch_size
        dloss_b = np.sum(G_batch, axis=1) / batch_size
        return dloss_W, dloss_b

    def propagate_G_batch(self, G_batch, weight, input):
        G_batch = weight.transpose().dot(G_batch)
        G_batch = G_batch * np.where(input > 0, input/input, input*0)
        return G_batch

    '''Remember here that the biases actually are actually the last column of the weight matrix'''

    def get_weights(self):
        weights_1 = np.delete(self.weights_1, np.size(self.weights_1, axis=1) - 1, 1)
        weights_2 = np.delete(self.weights_2, np.size(self.weights_2, axis=1) - 1, 1)
        return weights_1, weights_2

    def get_bias(self):
        bias_1 = np.array(self.weights_1[:, np.size(self.weights_1, axis=1) - 1])
        bias_2 = np.array(self.weights_2[:, np.size(self.weights_2, axis=1) - 1])
        return bias_1, bias_2

    def MiniBatchGD(self, train_data, val_data, penalty_factor, GDparams):
        batch_size, eta, n_epochs = GDparams

        #init information
        train_cost = np.zeros(n_epochs)
        validation_cost = np.zeros(n_epochs)
        train_loss = np.zeros(n_epochs)
        validation_loss = np.zeros(n_epochs)

        X, Y = train_data
        X_val, Y_val = val_data

        for i in range(n_epochs):
            self.fit(X, Y, penalty_factor, eta, batch_size)
            train_cost[i] = self.compute_cost(X, Y, penalty_factor)
            validation_cost[i] = self.compute_cost(X_val, Y_val, penalty_factor)
            train_loss[i] = self.compute_total_loss(X, Y)
            validation_loss[i] = self.compute_total_loss(X_val, Y_val)

        cost = train_cost, validation_cost
        loss = train_loss, validation_loss
        return cost, loss

    def fit(self, X, Y, penalty_factor, eta, batchSize=-1):

        if (batchSize == -1):
            batchSize = 1
            
        for i in range(0, X.shape[1], batchSize):
            batchX = X[:, i:i + batchSize]
            batchY = Y[:, i:i + batchSize]
            batchP = self.evaluate_classifier(batchX)
            [grad_W, grad_b] = self.compute_gradients(batchX, batchY, batchP, penalty_factor, batchSize)
            weights = self.get_weights() - eta * grad_W
            bias = self.get_bias() - eta * grad_b
            self.weights = np.column_stack((weights, bias))


def load_batch(filename):
    dict = LoadBatch(filename)
    X = np.array(dict[b'data'])
    y = np.array(dict[b'labels'])
    Y = make_one_hot_encoding(len(X), y)

    return [np.array(X.astype(float)).transpose(), np.array(Y.astype(float)).transpose(), y.astype(float)]


def make_one_hot_encoding(batch_size, indices):
    Y = np.zeros((batch_size, 10))
    for i in range(batch_size):
        hot = indices[i]
        Y[i][hot] = 1

    return Y


def pre_process(training_data):
    [X, Y, y] = training_data
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, ddof=1, axis=0)

    X = X - np.tile(mean_X, (np.size(X, axis=0), 1))
    X = X / np.tile(std_X, (np.size(X, axis=0), 1))

    return [X, Y, y]


'''def getDataSize(W):
    output_size = np.size(W, axis=0)
    input_size = np.size(W, axis=1)
    return input_size, output_size
'''

def ComputeCost(X, Y, neural_net, penalty_factor):
    return neural_net.compute_cost(X, Y, penalty_factor)


def ComputeGradients(X, Y, neural_net, penalty_factor, batch_size):
    batch_X = np.array(X[:, 0:0 + batch_size])
    batch_Y = np.array(Y[:, 0:0 + batch_size])
    P_batch = np.array(neural_net.evaluate_classifier(batch_X))
    batch_hidden = neural_net.hidden_layer_batch # We have evaluated the network so the values of the hidden layer should be available


    grad_analytiaclly = neural_net.compute_gradients(batch_X, batch_Y, P_batch, penalty_factor, batch_size)
    w1, w2 = neural_net.get_weights()
    b1, b2 = neural_net.get_bias()
    grad_numerically_1 = ComputeGradsNumSlow(batch_X, batch_hidden, w1, b1, penalty_factor, 0.00001)
    grad_numerically_2 = ComputeGradsNumSlow(batch_hidden, batch_Y, w2, b2, penalty_factor, 0.00001)

    return grad_analytiaclly, [grad_numerically_1, grad_numerically_2]


def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
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
    return numerator / denominator


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


def plot_cost(train_loss, val_loss):
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


def plot_total_loss(train_loss, val_loss):
    plt.plot(np.arange(np.size(train_loss)), train_loss, color='blue', label='Training Loss')
    plt.plot(np.arange(np.size(val_loss)), val_loss, color='red', label='Validation Loss')

    xMin = 0
    xMax = train_loss.size

    yMin = np.min(np.concatenate((train_loss, val_loss)))
    yMax = np.max(np.concatenate((train_loss, val_loss)))

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)

    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.show()


def plot_weight_matrix(weight_matrix):
    montage(weight_matrix)


if __name__ == '__main__':
    training_data = load_batch('data_batch_1')
    validation_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')

    processed_training_data = pre_process(training_data)
    processed_validation_data = pre_process(validation_data)
    processed_test_data = pre_process(test_data)

    output_size = np.size(processed_training_data[1], axis=0)
    input_size = np.size(processed_training_data[0], axis=0)
    hidden_size = 4000
    neural_net = ANN_two_layer(input_size, hidden_size, output_size)

    tdi = processed_training_data[0]
    tdl = processed_training_data[1]
    vdi = processed_validation_data[0]
    vdl = processed_validation_data[1]

    batch_size = 100
    eta = 0.1
    n_epochs = 40
    GDparams = batch_size, eta, n_epochs
    penalty_factor = 0

    # train_cost, validation_cost = neural_net.MiniBatchGD(tdi, tdl, vdi, vdl, penalty_factor, GDparams)
    # plot_cost(train_loss, validation_loss)

    train_loss, validation_loss = neural_net.MiniBatchGD(tdi, tdl, vdi, vdl, penalty_factor, GDparams)
    plot_total_loss(train_loss, validation_loss)
    #plot_weight_matrix(neural_net.get_weights())

    print("-----------------------------")
    print("Final train loss: ",
          neural_net.compute_cost(processed_training_data[0], processed_training_data[1], penalty_factor))
    print("Final validation loss: ",
          neural_net.compute_cost(processed_validation_data[0], processed_validation_data[1], penalty_factor))
    print("Final test loss: ", neural_net.compute_cost(processed_test_data[0], processed_test_data[1], penalty_factor))

    print("------------------------------")
    print("Final train accuracy: ", neural_net.compute_accuracy(processed_training_data[0], processed_training_data[2]))
    print("Final validation accuracy: ",
          neural_net.compute_accuracy(processed_validation_data[0], processed_validation_data[2]))
    print("Final test accuracy: ", neural_net.compute_accuracy(processed_test_data[0], processed_test_data[2]))

    '''neural_net = NeuralNet(input_size, output_size)
    penalty_factor = 0
    batch_size = 100
    grad_analytically, grad_numerically = ComputeGradients(processed_training_data[0], processed_training_data[1],
                                                           penalty_factor, batch_size)
    printOutGradients(grad_analytically, grad_numerically)

    eps = 0.001
    [grad_W, grad_b] = grad_analytically
    [grad_Wn, grad_bn] = grad_numerically

    print_gradient_check(grad_W, grad_Wn, grad_b, grad_bn, eps)'''
