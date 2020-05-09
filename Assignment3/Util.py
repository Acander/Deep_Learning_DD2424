from Assignment1.functions import LoadBatch
from Assignment1.functions import montage
import Assignment3.ANN_multilayer
import numpy as np
import matplotlib.pyplot as plt

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


def ComputeGradients(X, Y, neural_net, batch_size):
    batch_X = np.array(X[:, 0:0 + batch_size])
    batch_Y = np.array(Y[:, 0:0 + batch_size])
    P_batch = np.array(neural_net.evaluate_classifier(batch_X))

    grad_analytiaclly = neural_net.compute_gradients(batch_X, batch_Y, P_batch, batch_size)
    grad_numerically = ComputeGradsNumSlow(batch_X, batch_Y, neural_net, 0.00005)

    return grad_analytiaclly, grad_numerically


def ComputeGradsNumSlow(X, Y, neuarl_net, h):
    """ Converted from matlab code """
    # Note that W and B are arrays with the weights and biases for each layer kept seperatly
    W = neural_net.weights
    B = neural_net.biases
    W_size = np.size(W)
    B_size = np.size(B)

    grad_W = []
    grad_b = []

    neural_net_try = neural_net

    for j in range(B_size):
        grad_b.append(np.zeros(np.size(B[j])))
        for i in range(np.size(B[j])):
            b_try = B
            b_try[j][i] = b_try[j][i] - h
            neural_net_try.b = b_try
            c1, loss = neural_net_try.compute_cost_and_loss(X, Y)

            b_try = B
            b_try[j][i] = b_try[j][i] + h
            neural_net_try.b = b_try
            c2, loss = neural_net_try.compute_cost_and_loss(X, Y)

            grad_b[j][i] = (c2 - c1) / (2 * h)

    neural_net_try = neural_net

    for j in range(W_size):
        grad_W.append(np.zeros(np.shape(W[j])))
        # print(np.size(W[j], axis=0))
        for i in range(np.size(W[j], axis=0)):
            # print(np.size(W[j], axis=1))
            for k in range(np.size(W[j], axis=1)):
                W_try = W
                W_try[j][i][k] = W_try[j][i][k] - h
                neural_net_try.w = W_try
                c1, loss = neural_net_try.compute_cost_and_loss(X, Y)

                W_try = W
                W_try[j][i][k] = W_try[j][i][k] + h
                neural_net_try.w = W_try
                c2, loss = neural_net_try.compute_cost_and_loss(X, Y)

                grad_W[j][i][k] = (c2 - c1) / (2 * h)

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

    for i in range(np.size(grad_W)):
        print("Analytical (my own):.............", i)
        print(np.size(grad_W[i], axis=1))
        print(np.size(grad_b[i]))
        print(grad_W[i])
        print(grad_b[i])

    for i in range(np.size(grad_W)):
        print("Numerical (approximate):................", i)
        print(np.size(grad_Wn[i], axis=1))
        print(np.size(grad_bn[i]))
        print(grad_Wn[i])
        print(grad_bn[i])


def print_gradient_check(grad_W, grad_Wn, grad_b, grad_bn, eps):
    print("Gradient differences:")
    print("Weights:")
    for k in range(np.size(grad_W, axis=0)):
        for i in range(np.size(grad_W[k], axis=0)):
            for j in range(np.size(grad_W[k], axis=1)):
                print(gradient_difference(grad_W[k][i][j], grad_Wn[k][i][j], eps))

    print("Bias:")
    for k in range(np.size(grad_W, axis=0)):
        for i in range(np.size(grad_b[k], axis=0)):
            print(gradient_difference(grad_b[k][i], grad_bn[k][i], eps))


def plot_cost(train_cost, val_cost):
    plt.plot(np.arange(np.size(train_cost)), train_cost, color='blue', label='Training')
    plt.plot(np.arange(np.size(val_cost)), val_cost, color='red', label='Validation')

    xMin = 0
    xMax = train_cost.size

    yMin = np.min(np.concatenate((train_cost, val_cost)))
    yMax = np.max(np.concatenate((train_cost, val_cost)))

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)

    plt.xlabel('Update Step')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()


def plot_total_loss(train_loss, val_loss):
    plt.plot(np.arange(np.size(train_loss)), train_loss, color='blue', label='Training')
    plt.plot(np.arange(np.size(val_loss)), val_loss, color='red', label='Validation')

    xMin = 0
    xMax = train_loss.size

    yMin = np.min(np.concatenate((train_loss, val_loss)))
    yMax = np.max(np.concatenate((train_loss, val_loss)))

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)

    plt.xlabel('Update Step')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.show()


def plot_weight_matrix(weight_matrix):
    montage(weight_matrix)