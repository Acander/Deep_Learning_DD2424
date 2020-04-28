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
        self.eta_min, self.eta_max, self.step_size, self.n_cycles = eta_params
        self.t = 1


    def evaluate_classifier(self, X):
        hidden_layer = np.maximum(self.compute_hidden(X), np.zeros((self.hidden_size, np.size(X, axis=1))))
        self.hidden_layer_batch = hidden_layer
        P = softmax(self.compute_output(hidden_layer))
        return P


    def compute_hidden(self, X):
        sum_matrix = np.ones((1, np.size(X, axis=1)))
        S_1 = self.w[0].dot(X) + self.b[0].dot(sum_matrix)
        return S_1


    def compute_output(self, S_1):
        sum_matrix = np.ones((1, np.size(S_1, axis=1)))
        S = self.w[1].dot(S_1) + self.b[1].dot(sum_matrix)
        return S

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
        return [first_layer, second_layer]

    def init_G_batch(self, Y_batch, P_batch):
        return np.array(-(Y_batch - P_batch))

    def get_weight_gradient(self, layer_input_batch, G_batch, batch_size):
        dloss_W = G_batch.dot(layer_input_batch.transpose()) / batch_size
        dloss_b = G_batch.dot(np.ones((batch_size, 1))) / batch_size
        return dloss_W, dloss_b

    def propagate_G_batch(self, G_batch, weight, input):
        G_batch = weight.transpose().dot(G_batch)
        G_batch = G_batch * np.where(input > 0, input/input, input*0)
        return G_batch

    def MiniBatchGD(self, train_data, val_data, GDparams):
        batch_size, epochs = GDparams

        #init information
        train_cost = []
        validation_cost = []
        train_loss = []
        validation_loss = []

        X, Y = train_data
        X_val, Y_val = val_data

        for i in range(epochs):
            self.fit(X, Y, batch_size)
            if self.checkIfTrainingShouldStop():
                print("Should stop")
                break
            train_cost.append(self.compute_cost(X, Y))
            validation_cost.append(self.compute_cost(X_val, Y_val))
            train_loss.append(self.compute_total_loss(X, Y))
            validation_loss.append(self.compute_total_loss(X_val, Y_val))

        cost = train_cost, validation_cost
        loss = train_loss, validation_loss
        return cost, loss

    def fit(self, X, Y, batchSize=-1):

        if (batchSize == -1):
            batchSize = 1

        for i in range(0, X.shape[1], batchSize):
            eta_t = self.updatedLearningRate()
            if self.checkIfTrainingShouldStop():
                print("Should stop")
                print(self.t)
                break
            batchX = X[:, i:i + batchSize]
            batchY = Y[:, i:i + batchSize]
            batchP = self.evaluate_classifier(batchX)

            first_layer, second_layer = self.compute_gradients(batchX, batchY, batchP, batchSize)

            self.update_weights(first_layer, second_layer, eta_t)
            self.t += 1

        '''eta_t = self.updatedLearningRate()
        if self.checkIfTrainingShouldStop():
            print("Should stop")
            print(self.t)
            return 0
        batchX = X[:, 0:batchSize]
        batchY = Y[:, 0:batchSize]
        batchP = self.evaluate_classifier(batchX)

        first_layer, second_layer = self.compute_gradients(batchX, batchY, batchP, batchSize)

        self.update_weights(first_layer, second_layer, eta_t)
        self.t += 1'''

    def update_weights(self, first_layer_gradient, second_layer_gradient, eta):
        gradient_W1, gradient_b1 = first_layer_gradient
        gradient_W2, gradient_b2 = second_layer_gradient

        gradient_b1 = np.reshape(gradient_b1, (np.size(gradient_b1), 1))
        gradient_b2 = np.reshape(gradient_b2, (np.size(gradient_b2), 1))

        self.w[0] = self.w[0] - eta * gradient_W1
        self.w[1] = self.w[1] - eta * gradient_W2
        self.b[0] = self.b[0] - eta * gradient_b1
        self.b[1] = self.b[1] - eta * gradient_b2

    def updatedLearningRate(self):
        l = np.floor(self.t / (2 * self.step_size))
        if 2*l*self.step_size <= self.t < (2 * l + 1)*self.step_size:
            return self.evenIterationFunc(l, self.t)
        else:
            return self.unevenIterationFunc(l, self.t)

    def evenIterationFunc(self, l, t):
        return self.eta_min + (t - 2*l*self.step_size)/self.step_size*(self.eta_max - self.eta_min)

    def unevenIterationFunc(self, l, t):
        return self.eta_max - (t - (2*l+1)*self.step_size)/self.step_size*(self.eta_max - self.eta_min)

    def checkIfTrainingShouldStop(self):
        return self.n_cycles*2 == self.t/self.step_size

def load_batch(filename):
    dict = LoadBatch(filename)
    X = np.array(dict[b'data'])
    y = np.array(dict[b'labels'])
    Y = make_one_hot_encoding(len(X), y)

    return [np.array(X.astype(float)).transpose(), np.array(Y.astype(float)).transpose(), y.astype(float)]

'''def load_batches(filenames):
    X, y, Y = [], [], []
    for i, filename in enumerate(filenames):
        dict = LoadBatch(filename)
        X = np.concatenate((X, dict[b'data']))
        y = np.concatenate((y, dict[b'labels']))
        Y = np.concatenate((Y, make_one_hot_encoding(len(X), y)))

    return [np.array(X.astype(float)).transpose(), Y.astype(float).transpose(), y.astype(float)]'''


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
    #Note that W and B are arrays with the weights and biases for each layer kept seperatly
    W = neural_net.w
    B = neural_net.b
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
            c1 = neural_net_try.compute_cost(X, Y)

            b_try = B
            b_try[j][i] = b_try[j][i] + h
            neural_net_try.b = b_try
            c2 = neural_net_try.compute_cost(X, Y)

            grad_b[j][i] = (c2 - c1) / (2 * h)

    neural_net_try = neural_net

    for j in range(W_size):
        grad_W.append(np.zeros(np.shape(W[j])))
        #print(np.size(W[j], axis=0))
        for i in range(np.size(W[j], axis=0)):
            #print(np.size(W[j], axis=1))
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
    for k in range(np.size(grad_W, axis=0)):
        for i in range(np.size(grad_W[k], axis=0)):
            for j in range(np.size(grad_W[k], axis=1)):
                print(gradient_difference(grad_W[k][i][j], grad_Wn[k][i][j], eps))

    print("Bias:")
    for k in range(np.size(grad_W, axis=0)):
        for i in range(np.size(grad_b[k], axis=0)):
            print(gradient_difference(grad_b[k][i], grad_bn[k][i], eps))


def plot_cost(train_cost, val_cost):
    plt.plot(np.arange(np.size(train_cost)), train_cost, color='blue', label='Training Loss')
    plt.plot(np.arange(np.size(val_cost)), val_cost, color='red', label='Validation Loss')

    xMin = 0
    xMax = train_cost.size

    yMin = np.min(np.concatenate((train_cost, val_cost)))
    yMax = np.max(np.concatenate((train_cost, val_cost)))

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
    [X_train_1, Y_train_1, y_train_1] = load_batch('data_batch_1')
    [X_train_2, Y_train_2, y_train_2] = load_batch('data_batch_2')
    [X_train_3, Y_train_3, y_train_3] = load_batch('data_batch_3')
    [X_train_4, Y_train_4, y_train_4] = load_batch('data_batch_4')

    X_train = np.concatenate((X_train_1, X_train_2), axis=1)
    X_train = np.concatenate((X_train, X_train_3), axis=1)
    X_train = np.concatenate((X_train, X_train_4), axis=1)

    Y_train = np.concatenate((Y_train_1, Y_train_2), axis=1)
    Y_train = np.concatenate((Y_train, Y_train_3), axis=1)
    Y_train = np.concatenate((Y_train, Y_train_4), axis=1)

    y_train = np.concatenate((y_train_1, y_train_2))
    y_train = np.concatenate((y_train, y_train_3))
    y_train = np.concatenate((y_train, y_train_4))

    training_data = [X_train, Y_train, y_train]
    validation_data = load_batch('data_batch_5')
    test_data = load_batch('test_batch')

    processed_training_data = pre_process(training_data)
    processed_validation_data = pre_process(validation_data)
    processed_test_data = pre_process(test_data)

    output_size = np.size(processed_training_data[1], axis=0)
    input_size = np.size(processed_training_data[0], axis=0)
    hidden_size = 50
    #lamda = 0.01
    eta_min = 0.00001
    eta_max = 0.1
    step_size = 500
    n_cycles = 2
    #eta_params = eta_min, eta_max, step_size, n_cycles
    #neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamda, eta_params)
    neural_net = 0

    batch_size = 100
    epochs = 1000
    GDparams = batch_size, epochs

    tdi = processed_training_data[0]
    tdl = processed_training_data[1]
    vdi = processed_validation_data[0]
    vdl = processed_validation_data[1]
    
    train_data = tdi, tdl
    val_data = vdi, vdl

    iterations = 10
    best_lamda = 0
    high_val_acc = 0
    l_min = -5
    l_max = -1
    l = l_min + (l_max - l_min) * np.random.uniform(0, 1, iterations)
    print(l)
    lamdas = 10 ** l
    print(lamdas)
    step_size = 2 * np.floor(np.size(processed_training_data[0], axis=1) / batch_size)
    eta_params = eta_min, eta_max, step_size, n_cycles

    #Course search
    for i in range(iterations):
        neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamdas[i], eta_params)
        neural_net.MiniBatchGD(train_data, val_data, GDparams)
        val_accuracy = neural_net.compute_accuracy(processed_validation_data[0], processed_validation_data[2])
        if val_accuracy > high_val_acc:
            high_val_acc = val_accuracy
            best_lamda = i
            print("Best lamda: ", lamdas[best_lamda])
            print("Val_accuracy: ", high_val_acc)

    lamdas = lamdas[best_lamda] + np.random.uniform(-0.01, 0.01, iterations)
    iterations = 10
    best_lamda = 0
    high_val_acc = 0

    # Narrow search
    for i in range(iterations):
        neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamdas[i], eta_params)
        neural_net.MiniBatchGD(train_data, val_data, GDparams)
        val_accuracy = neural_net.compute_accuracy(processed_validation_data[0], processed_validation_data[2])
        if val_accuracy > high_val_acc:
            high_val_acc = val_accuracy
            best_lamda = i
            print("Best lamda: ", lamdas[best_lamda])
            print("Val_accuracy: ", high_val_acc)

    neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamdas[best_lamda], eta_params)
    neural_net.MiniBatchGD(train_data, val_data, GDparams)

    '''
    cost, loss = neural_net.MiniBatchGD(train_data, val_data, GDparams)
    train_cost, validation_cost = cost
    train_loss, validation_loss = loss
    plot_cost(np.array(train_cost), validation_cost)
    plot_total_loss(np.array(train_loss), validation_loss)'''

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
    print("Final test accuracy: ", neural_net.compute_accuracy(processed_test_data[0], processed_test_data[2]))


    '''grad_analytically, grad_numerically = ComputeGradients(processed_training_data[0], processed_training_data[1], neural_net,
                                                           batch_size)
    printOutGradients(grad_analytically, grad_numerically)

    eps = 0.001
    [grad_W, grad_b] = grad_analytically
    [grad_Wn, grad_bn] = grad_numerically

    print_gradient_check(grad_W, grad_Wn, grad_b, grad_bn, eps)'''
