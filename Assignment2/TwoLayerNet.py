from Assignment1.functions import softmax
from Assignment1.functions import LoadBatch
from Assignment1.functions import montage
import numpy as np
import matplotlib.pyplot as plt

class ANN_two_layer:

    def __init__(self, input_size, hidden_size, output_size, lamda, eta_params, weights_1=0, bias_1=0, weights_2=0, bias_2=0):
        mu, sigma = 0, 0.01
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lamda = lamda
        self.hidden_layer_batch = np.matrix((hidden_size, 1))
        self.w = []
        self.b = []

        if weights_1 is 0:
            self.w.append(np.random.normal(mu, sigma, (hidden_size, input_size)))
            self.w.append(np.random.normal(mu, sigma, (output_size, hidden_size)))
            self.b.append(np.zeros((hidden_size, 1)))
            self.b.append(np.zeros((output_size, 1)))
        else:
            # print("I ran!")
            self.w.append(np.matrix(weights_1))
            self.w.append(np.matrix(weights_2))
            self.b.append(np.array(bias_1).reshape((hidden_size, 1)))
            self.b.append(np.array(bias_2).reshape((output_size, 1)))

        #Paramters related to cyclic learning rate:
        self.eta_min, self.eta_max, self.step_size = eta_params

    '''def bunch_up_params(self):
        weight_matrices = np.array([self.w[0], self.w[1]])
        bias_arrays = np.array([self.b[0], self.b[1]])
        return weight_matrices, bias_arrays'''

    def evaluate_classifier(self, X):
        hidden_layer = np.maximum(self.compute_hidden(X), np.zeros((self.hidden_size, np.size(X, axis=1))))
        self.hidden_layer_batch = hidden_layer
        P = softmax(self.compute_output(hidden_layer))
        return P

    '''def compute_hidden(self, X):
        b1, b2 = self.get_bias()
        b1 = np.matrix(b1).transpose()
        sum_matrix = np.matrix(np.ones(np.size(X, axis=1)))
        w1, w2 = self.get_weights()
        S_1 = w1.dot(X) + b1.dot(sum_matrix)
        return S_1'''

    def compute_hidden(self, X):
        sum_matrix = np.ones((1, np.size(X, axis=1)))
        S_1 = self.w[0].dot(X) + self.b[0].dot(sum_matrix)
        return S_1

    '''def compute_output(self, S_1):
        b1, b2 = self.get_bias()
        b1 = np.matrix(b1).transpose()
        sum_matrix = np.matrix(np.ones(np.size(S_1, axis=1)))
        w1, w2 = self.get_weights()
        S = w1.dot(S_1) + b1.dot(sum_matrix)
        return S'''

    def compute_output(self, S_1):
        sum_matrix = np.ones((1, np.size(S_1, axis=1)))
        S = self.w[1].dot(S_1) + self.b[1].dot(sum_matrix)
        return S

    '''def compute_cost(self, X, Y):
        norm_factor = 1 / np.size(X, axis=1)
        P = self.evaluate_classifier(X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(P, axis=1)

        for i in range(np.size(Y, axis=1)):
            sum_entropy += self.cross_entropy(P[:, i], Y[:, i])

        w1, w2 = self.get_weights()
        penalty_term = self.lamda * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return norm_factor * sum_entropy + penalty_term'''

    def compute_cost(self, X, Y):
        norm_factor = 1 / np.size(X, axis=1)
        P = self.evaluate_classifier(X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(P, axis=1)

        for i in range(np.size(Y, axis=1)):
            sum_entropy += self.cross_entropy(P[:, i], Y[:, i])

        penalty_term = self.lamda * (np.sum(np.square(self.w[0])) + np.sum(np.square(self.w[1])))
        return norm_factor * sum_entropy + penalty_term

    def cross_entropy(self, p, y):
        # s: softmax network output
        # y: expected output - one-hot encoding
        p = np.array(p).reshape((np.size(p), 1))
        y = np.array(y).reshape((1, np.size(y)))
        return -np.log10(y.dot(p)[0][0])

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

    '''def compute_gradients(self, X_batch, Y_batch, P_batch, penalty_factor, batch_size):
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

        first_layer = gradient_W1, gradient_b1
        second_layer = gradient_W2, gradient_b2
        return first_layer, second_layer'''

    def compute_gradients(self, X_batch, Y_batch, P_batch, batch_size):
        # We backprop the gradient G through the net #
        G_batch = self.init_G_batch(Y_batch, P_batch)

        dloss_W2, dloss_b2 = self.get_weight_gradient(self.hidden_layer_batch, G_batch, batch_size)
        G_batch = self.propagate_G_batch(G_batch, self.w[1], self.hidden_layer_batch)
        dloss_W1, dloss_b1 = self.get_weight_gradient(X_batch, G_batch, batch_size)

        gradient_W1 = dloss_W1 + 2 * self.lamda * self.w[0]
        gradient_b1 = dloss_b1

        gradient_W2 = dloss_W2 + 2 * self.lamda * self.w[1]
        gradient_b2 = dloss_b2

        first_layer = gradient_W1, gradient_b1
        second_layer = gradient_W2, gradient_b2
        grad_W = np.array([gradient_W1, gradient_W2])
        grad_b = np.array([gradient_b1, gradient_b2])
        return [grad_W, grad_b]

    def init_G_batch(self, Y_batch, P_batch):
        return np.array(-(Y_batch - P_batch))

    def get_weight_gradient(self, layer_input_batch, G_batch, batch_size):
        dloss_W = G_batch.dot(layer_input_batch.transpose()) / batch_size
        dloss_b = np.sum(G_batch, axis=1) / batch_size
        return dloss_W, dloss_b

    def propagate_G_batch(self, G_batch, weight, input):
        '''print(G_batch)
        print(weight)
        print(input)'''
        G_batch = weight.transpose().dot(G_batch)
        G_batch = G_batch * np.where(input > 0, input/input, input*0)
        return G_batch

    '''Remember here that the biases actually are actually the last column of the weight matrix'''

    '''def get_weights(self):
        weights_1 = np.delete(self.weights_1, np.size(self.weights_1, axis=1) - 1, 1)
        weights_2 = np.delete(self.weights_2, np.size(self.weights_2, axis=1) - 1, 1)
        return weights_1, weights_2

    def get_bias(self):
        bias_1 = np.array(self.weights_1[:, np.size(self.weights_1, axis=1) - 1])
        bias_2 = np.array(self.weights_2[:, np.size(self.weights_2, axis=1) - 1])
        return bias_1, bias_2'''

    def MiniBatchGD(self, train_data, val_data, GDparams):
        batch_size, n_cycles = GDparams

        #init information
        train_cost = np.zeros(n_cycles)
        validation_cost = np.zeros(n_cycles)
        train_loss = np.zeros(n_cycles)
        validation_loss = np.zeros(n_cycles)

        X, Y = train_data
        X_val, Y_val = val_data

        t = 0 #Time step for cyclic learning rate

        for i in range(n_cycles):
            eta_t = self.updatedLearningRate(t, n_cycles)
            self.fit(X, Y, eta_t, batch_size)
            train_cost[i] = self.compute_cost(X, Y)
            validation_cost[i] = self.compute_cost(X_val, Y_val)
            train_loss[i] = self.compute_total_loss(X, Y)
            validation_loss[i] = self.compute_total_loss(X_val, Y_val)

        cost = train_cost, validation_cost
        loss = train_loss, validation_loss
        return cost, loss

    def fit(self, X, Y, eta, batchSize=-1):

        if (batchSize == -1):
            batchSize = 1

        for i in range(0, X.shape[1], batchSize):
            batchX = X[:, i:i + batchSize]
            batchY = Y[:, i:i + batchSize]
            batchP = self.evaluate_classifier(batchX)

            first_layer, second_layer = self.compute_gradients(batchX, batchY, batchP, batchSize)

            self.update_weights(first_layer, second_layer, eta)


    def update_weights(self, first_layer_gradient, second_layer_gradient, eta):
        gradient_W1, gradient_b1 = first_layer_gradient
        gradient_W2, gradient_b2 = second_layer_gradient

        self.w[0] = self.w[0] - eta * gradient_W1
        self.w[1] = self.w[1] - eta * gradient_W2
        self.b[0] = self.b[0] - eta * gradient_b1
        self.b[1] = self.b[1] - eta * gradient_b2

        #self.construct_weight_matrix(w1, b1)
        #self.construct_weight_matrix(w2, b2)

    def updatedLearningRate(self, t, n_cycles):
        steps = np.arange(n_cycles*2)
        intervals = steps*self.step_size
        for i in range(intervals):
            if (intervals[i] <= t < intervals):
                if i % 2 > 0:
                    return self.unevenIterationFunc(i, t)
                else:
                    return self.evenIterationFunc(i, t)

    def evenIterationFunc(self, l, t):
        return self.eta_min + (t - 2*l*self.step_size)/self.step_size*(self.eta_max - self.eta_min)

    def unevenIterationFunc(self, l, t):
        return self.eta_max - (t - 2*l*self.step_size)/self.step_size*(self.eta_max - self.eta_min)

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

'''def ComputeCost(X, Y, neural_net, penalty_factor):
    return neural_net.compute_cost(X, Y, penalty_factor)
'''

def ComputeGradients(X, Y, neural_net, batch_size):
    batch_X = np.array(X[:, 0:0 + batch_size])
    batch_Y = np.array(Y[:, 0:0 + batch_size])
    P_batch = np.array(neural_net.evaluate_classifier(batch_X))

    grad_analytiaclly = neural_net.compute_gradients(batch_X, batch_Y, P_batch, batch_size)
    grad_numerically = ComputeGradsNumSlow(batch_X, batch_Y, neural_net, 0.00005)

    return grad_analytiaclly, grad_numerically


def ComputeGradsNumSlow(X, Y, neuarl_net, h):
    """ Converted from matlab code """
    #Note that W and B are arrays with the weights and biases for each layer kept seperatly
    W = neural_net.w
    B = neural_net.b
    W_size = np.size(W)
    B_size = np.size(B)

    '''#print(grad_W)
    #print(grad_b)

    #Init grad vectors for bias and weights
    for i in range(W_size):
        grad_W.append(np.zeros(np.size(W[i])))

    for i in range(B_size):
        grad_b.append(np.zeros(np.size(B[i])))
    '''

    grad_W = []
    grad_b = []

    neural_net_try = neural_net

    for j in range(B_size):
        grad_b.append(np.zeros(np.size(B[j])))
        for i in range(np.size(B[j])):
            b_try = B
            b_try[j][i] = b_try[j][i] - h
            neural_net_try.b = b_try
            c1 = neural_net_try.compute_cost(X, Y)

            b_try = B
            b_try[j][i] = b_try[j][i] + h
            neural_net_try.b = b_try
            c2 = neural_net_try.compute_cost(X, Y)

            #print(c2)
            #print(c1)
            grad_b[j][i] = (c2 - c1) / (2 * h)

    neural_net_try = neural_net

    for j in range(W_size):
        grad_W.append(np.zeros(np.shape(W[j])))
        print(np.size(W[j], axis=0))
        for i in range(np.size(W[j], axis=0)):
            print(np.size(W[j], axis=1))
            for k in range(np.size(W[j], axis=1)):
                W_try = W
                W_try[j][i][k] = W_try[j][i][k] - h
                neural_net_try.w = W_try
                c1 = neural_net_try.compute_cost(X, Y)

                W_try = W
                W_try[j][i][k] = W_try[j][i][k] + h
                neural_net_try.w = W_try
                c2 = neural_net_try.compute_cost(X, Y)

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
    for k in range(np.size(grad_W)):
        for i in range(np.size(grad_W[k], axis=0)):
            for j in range(np.size(grad_W[k], axis=1)):
                print(gradient_difference(grad_W[k][i][j], grad_Wn[k][i][j], eps))

    print("Bias:")
    for k in range(np.size(grad_W)):
        for i in range(np.size(grad_b[k], axis=0)):
            print(gradient_difference(grad_b[k][i], grad_bn[k][i], eps))


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
    hidden_size = 2
    lamda = 0
    eta_min = 0.00005
    eta_max = 0.1
    step_size = 500
    eta_params = eta_min, eta_max, step_size
    neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamda, eta_params)

    '''batch_size = 100
    n_cycles = 40
    GDparams = batch_size, n_cycles

    tdi = processed_training_data[0]
    tdl = processed_training_data[1]
    vdi = processed_validation_data[0]
    vdl = processed_validation_data[1]
    
    train_data = tdi, tdl
    val_data = vdi, vdl
    
    cost, loss = neural_net.MiniBatchGD(train_data, val_data, GDparams)
    train_cost, validation_cost = cost
    train_loss, validation_loss = loss
    plot_cost(train_loss, validation_loss)
    plot_total_loss(train_loss, validation_loss)
    
    #plot_weight_matrix(neural_net.get_weights())

    print("-----------------------------")
    print("Final train loss: ",
          neural_net.compute_cost(processed_training_data[0], processed_training_data[1]))
    print("Final validation loss: ",
          neural_net.compute_cost(processed_validation_data[0], processed_validation_data[1]))
    print("Final test loss: ", neural_net.compute_cost(processed_test_data[0], processed_test_data[1]))

    print("------------------------------")
    print("Final train accuracy: ", neural_net.compute_accuracy(processed_training_data[0], processed_training_data[2]))
    print("Final validation accuracy: ",
          neural_net.compute_accuracy(processed_validation_data[0], processed_validation_data[2]))
    print("Final test accuracy: ", neural_net.compute_accuracy(processed_test_data[0], processed_test_data[2]))'''

    penalty_factor = 0
    batch_size = 100
    grad_analytically, grad_numerically = ComputeGradients(processed_training_data[0], processed_training_data[1], neural_net,
                                                           batch_size)
    printOutGradients(grad_analytically, grad_numerically)

    eps = 0.001
    [grad_W, grad_b] = grad_analytically
    [grad_Wn, grad_bn] = grad_numerically

    print_gradient_check(grad_W, grad_Wn, grad_b, grad_bn, eps)
