from Assignment1.functions import softmax
from Assignment1.functions import LoadBatch
from Assignment1.functions import montage
import numpy as np
import matplotlib.pyplot as plt

class ANN_multilayer:

    def __init__(self, layers, lamda, eta_params):
        '''The 'layers' parameter is an array of integers, each representing the size of a hidden layer.'''

        mu, sigma = 0, 0.01
        self.n_layers = len(layers) #Including input and output layers
        self.layers = layers

        self.lamda = lamda
        self.hidden_layers_batch = []
        self.weights = []
        self.biases = []

        for i in range(self.n_layers-1):
            self.weights.append(np.random.normal(mu, sigma, (layers[i+1], layers[i])))
            self.biases.append(np.zeros((layers[i+1], 1)))
            self.hidden_layers_batch.append(np.matrix((layers[i+1], 1)))

        # Parameters related to cyclic learning rate:
        self.eta_min, self.eta_max, self.step_size, self.n_cycles = eta_params
        self.t = 1


    def evaluate_classifier(self, X):
        S_l = X
        hidden_layer = 0
        for i in range(self.n_layers-2):
            hidden_index = i + 1
            S_l = self.compute_hidden(S_l, hidden_index)
            S_l = np.maximum(S_l, np.zeros((self.layers[hidden_index], np.size(X, axis=1))))
            self.hidden_layers_batch[hidden_index] = S_l
        P = softmax(self.compute_hidden(S_l, self.n_layers-1))
        return P


    def compute_hidden(self, X, layer):
        sum_matrix = np.ones((1, np.size(X, axis=1)))
        S_1 = self.weights[layer-1].dot(X) + self.biases[layer-1].dot(sum_matrix)
        return S_1

    def compute_cost_and_loss(self, X, Y):
        norm_factor = 1 / np.size(X, axis=1)
        P = self.evaluate_classifier(X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(P, axis=1)

        for i in range(np.size(Y, axis=1)):
            sum_entropy += self.cross_entropy(P[:, i], Y[:, i])

        penalty_term = self.lamda * (np.sum(np.square(self.w[0])) + np.sum(np.square(self.w[1])))
        cost = norm_factor * sum_entropy + penalty_term
        loss = norm_factor*sum_entropy
        return cost, loss

    def cross_entropy(self, p, y):
        # s: softmax network output
        # y: expected output - one-hot encoding
        return -np.log10(np.dot(np.array(y), p))

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
        gradients = []
        G_batch = self.init_G_batch(Y_batch, P_batch)

        n_hidden_layers = self.n_layers-2
        for l in range(n_hidden_layers):
            dloss_Wl, dloss_bl = self.get_weight_gradient(self.hidden_layers_batch[n_hidden_layers-l-1], G_batch, batch_size)
            G_batch = self.propagate_G_batch(G_batch, self.weights[n_hidden_layers-l-1], self.hidden_layers_batch[n_hidden_layers-l-1])
            gradient_Wl = dloss_Wl + 2 * self.lamda * self.weights[n_hidden_layers-l-1]
            gradient_bl = dloss_bl
            layer = gradient_Wl, gradient_bl
            gradients.append(layer)

        dloss_W1, dloss_b1 = self.get_weight_gradient(X_batch, G_batch, batch_size)
        gradient_W1 = dloss_W1 + 2 * self.lamda * self.weights[0]
        gradient_b1 = dloss_b1
        layer = gradient_W1, gradient_b1
        gradients.append(layer)
        return gradients.reverse()

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
            cost, loss = self.fit(X, Y, X_val, Y_val, batch_size)

            train_cost = np.concatenate((train_cost, cost[0]))
            validation_cost = np.concatenate((validation_cost, cost[1]))
            train_loss = np.concatenate((train_loss, loss[0]))
            validation_loss = np.concatenate((validation_loss, loss[1]))
            if self.checkIfTrainingShouldStop():
                break

        cost = train_cost, validation_cost
        loss = train_loss, validation_loss

        return cost, loss

    def fit(self, X, Y, X_val, Y_val, batchSize=-1):

        # init information
        train_cost = []
        val_cost = []
        train_loss = []
        val_loss = []

        if (batchSize == -1):
            batchSize = 1

        for i in range(0, X.shape[1], batchSize):
            #print(i)
            eta_t = self.updatedLearningRate()
            if self.checkIfTrainingShouldStop():
                #print(self.t)
                break
            batchX = X[:, i:i + batchSize]
            batchY = Y[:, i:i + batchSize]
            batchP = self.evaluate_classifier(batchX)

            gradients = self.compute_gradients(batchX, batchY, batchP, batchSize)

            self.update_weights(gradients, eta_t)
            self.t += 1

            '''if i % 1000 == 0:
                #print("Compute cost and loss")
                tc, tl = self.compute_cost_and_loss(X, Y)
                vc, vl = self.compute_cost_and_loss(X_val, Y_val)
                train_cost.append(tc)
                val_cost.append(vc)
                train_loss.append(tl)
                val_loss.append(vl)'''

        cost = [train_cost, val_cost]
        loss = [train_loss, val_loss]

        return cost, loss

    def update_weights(self, gradients, eta):
        for i, gradient in enumerate(gradients):
            gradient_Wl, gradient_bl = gradient
            gradient_bl = np.reshape(gradient_bl, (np.size(gradient_bl), 1))

            self.weights[i] = self.weights[i] - eta * gradient_Wl
            self.biases[i] = self.biases[i] - eta * gradient_bl

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
            c1, loss = neural_net_try.compute_cost_and_loss(X, Y)

            b_try = B
            b_try[j][i] = b_try[j][i] + h
            neural_net_try.b = b_try
            c2, loss = neural_net_try.compute_cost_and_loss(X, Y)

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


if __name__ == '__main__':
    [X_train_1, Y_train_1, y_train_1] = load_batch('data_batch_1')
    [X_train_2, Y_train_2, y_train_2] = load_batch('data_batch_2')
    [X_train_3, Y_train_3, y_train_3] = load_batch('data_batch_3')
    [X_train_4, Y_train_4, y_train_4] = load_batch('data_batch_4')
    [X_train_5, Y_train_5, y_train_5] = load_batch('data_batch_5')

    '''X_train_5, X_val = np.split(X_train_5, 2, axis=1)
    Y_train_5, Y_val = np.split(Y_train_5, 2, axis=1)
    y_train_5, y_val = np.split(y_train_5, 2)'''

    [X_train_5, X_val] = np.split(X_train_5, [9000], axis=1)
    [Y_train_5, Y_val] = np.split(Y_train_5, [9000], axis=1)
    [y_train_5, y_val] = np.split(y_train_5, [9000])

    X_train = np.concatenate((X_train_1, X_train_2), axis=1)
    X_train = np.concatenate((X_train, X_train_3), axis=1)
    X_train = np.concatenate((X_train, X_train_4), axis=1)
    X_train = np.concatenate((X_train, X_train_5), axis=1)

    Y_train = np.concatenate((Y_train_1, Y_train_2), axis=1)
    Y_train = np.concatenate((Y_train, Y_train_3), axis=1)
    Y_train = np.concatenate((Y_train, Y_train_4), axis=1)
    Y_train = np.concatenate((Y_train, Y_train_5), axis=1)

    y_train = np.concatenate((y_train_1, y_train_2))
    y_train = np.concatenate((y_train, y_train_3))
    y_train = np.concatenate((y_train, y_train_4))
    y_train = np.concatenate((y_train, y_train_5))

    training_data = [X_train, Y_train, y_train]
    validation_data = [X_val, Y_val, y_val]
    test_data = load_batch('test_batch')

    '''training_data = load_batch('data_batch_1')
    validation_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')'''

    processed_training_data = pre_process(training_data)
    processed_validation_data = pre_process(validation_data)
    processed_test_data = pre_process(test_data)

    output_size = np.size(processed_training_data[1], axis=0)
    input_size = np.size(processed_training_data[0], axis=0)
    layers = [input_size, 50, output_size]

    lamda = 0.0010995835253050919
    #lamda = 0
    eta_min = 0.00001
    eta_max = 0.1
    batch_size = 100
    #step_size = 800
    step_size = 2 * np.floor(np.size(processed_training_data[0], axis=1) / batch_size)

    n_cycles = 3
    eta_params = eta_min, eta_max, step_size, n_cycles
    neural_net = ANN_multilayer(layers, lamda, eta_params)

    epochs = 1000
    GDparams = batch_size, epochs

    tdi = processed_training_data[0]
    tdl = processed_training_data[1]
    vdi = processed_validation_data[0]
    vdl = processed_validation_data[1]
    
    train_data = tdi, tdl
    val_data = vdi, vdl
    #neural_net.MiniBatchGD(train_data, val_data, GDparams)

    '''#Lambda optimization
    ####################################################################

    iterations_c = 5
    #best_lamda = 0
    best_lamda = 0
    #high_val_acc = 0
    high_val_acc = 0.5208
    l_min = -5
    l_max = -1
    #l = l_min + (l_max - l_min) * np.random.uniform(0, 1, iterations_c)
    #print(l)
    #lamdas_course = 10 ** l
    lamdas_course = [1.10713074e-03, 2.43313844e-05, 7.50698134e-02, 2.67369941e-04, 1.12718786e-03]
    print(lamdas_course)
    step_size = 2 * np.floor(np.size(processed_training_data[0], axis=1) / batch_size)
    eta_params = eta_min, eta_max, step_size, n_cycles

    print("COURSE SEARCH")
    for i in range(iterations_c):
        print("Iteration: ", i)
        neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamdas_course[i], eta_params)
        neural_net.MiniBatchGD(train_data, val_data, GDparams)
        val_accuracy = neural_net.compute_accuracy(processed_validation_data[0], processed_validation_data[2])
        if val_accuracy > high_val_acc:
            high_val_acc = val_accuracy
            best_lamda = i
            print("Best lamda: ", lamdas_course[best_lamda])
            print("Val_accuracy: ", high_val_acc)

    margins_of_search = 0.00001
    lamdas = lamdas_course[best_lamda] + np.random.uniform(-margins_of_search, margins_of_search, iterations_c)
    lamdas = np.concatenate((lamdas_course, lamdas))
    iterations = 5

    #print("\n")

    print("FINE SEARCH")
    for i in range(iterations_c, iterations_c + iterations):
        print("Iteration: ", i)
        neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamdas[i], eta_params)
        neural_net.MiniBatchGD(train_data, val_data, GDparams)
        val_accuracy = neural_net.compute_accuracy(processed_validation_data[0], processed_validation_data[2])
        if val_accuracy > high_val_acc:
            high_val_acc = val_accuracy
            best_lamda = i
            print("Best lamda: ", lamdas[best_lamda])
            print("Val_accuracy: ", high_val_acc)

    #neural_net = ANN_two_layer(input_size, hidden_size, output_size, lamdas[best_lamda], eta_params)
    #neural_net.MiniBatchGD(train_data, val_data, GDparams)

    ############################################################################'''

    cost, loss = neural_net.MiniBatchGD(train_data, val_data, GDparams)
    '''train_cost, validation_cost = cost
    train_loss, validation_loss = loss
    plot_cost(np.array(train_cost), validation_cost)
    plot_total_loss(np.array(train_loss), validation_loss)

    print("Time steps performed: ", neural_net.t)
    print("Train Cost length: ", np.size(train_cost))

    #train_cost, train_loss = neural_net.compute_cost_and_loss(processed_training_data[0], processed_training_data[1])'''

    print("-----------------------------")
    print("Final train loss: ", )
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
