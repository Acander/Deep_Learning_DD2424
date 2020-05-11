import numpy as np
import sys
import sklearn
from Assignment1.functions import softmax
from Assignment1.functions import LoadBatch
from Assignment1.functions import montage
import matplotlib.pyplot as plt

eps = sys.float_info.epsilon

class ANN_multilayer:

    def __init__(self, layers, lamda, eta_params, BN=False, alfa=0):
        '''The 'layers' parameter is an array of integers, each representing the size of a hidden layer.'''

        #mu, sigma = 0, 0.01 # Xavier init
        #mu, sigma = 0, 2/layers[0] # He init
        self.n_layers = len(layers)  # Including input and output layers
        self.layers = layers
        self.BN = BN

        self.lamda = lamda
        self.alfa = alfa
        self.inputs_batch = []
        self.hidden_layers_batch = []
        self.hidden_layers_mod_batch = []
        self.final_prob_batch = []
        self.weights = []
        self.biases = []

        # Batch Normalization
        self.batch_means = []
        self.batch_variances = []
        self.batch_means_avg = []
        self.batch_variances_avg = []
        self.gammas = []
        self.betas = []

        for i in range(self.n_layers - 1):
            #self.weights.append(np.random.normal(mu, sigma, (layers[i + 1], layers[i]))) # Xavier init
            self.weights.append(np.random.normal(0, 1/np.sqrt(layers[i]), (layers[i + 1], layers[i]))) # He init
            #self.weights.append(np.random.normal(0, 1e-4, (layers[i + 1], layers[i])))
            self.biases.append(np.zeros((layers[i + 1], 1)))

        for i in range(self.n_layers-2):
            self.hidden_layers_batch.append(np.matrix((layers[i + 1], 1)))
            if BN:
                self.inputs_batch.append(np.matrix((layers[i + 1], 1)))
                self.hidden_layers_mod_batch.append(np.matrix((layers[i + 1], 1)))
                self.gammas.append(np.ones((layers[i + 1], 1)))
                self.betas.append(np.zeros((layers[i + 1], 1)))
                self.batch_means.append(np.array(layers[i + 1]))
                self.batch_variances.append(np.array(layers[i + 1]))
                self.batch_means_avg.append(np.array(layers[i + 1]))
                self.batch_variances_avg.append(np.array(layers[i + 1]))
                # self.networks_final_prob_batch.append(np.matrix((layers[i + 1], 1)))


        # Parameters related to cyclic learning rate:
        self.eta_min, self.eta_max, self.step_size, self.n_cycles = eta_params
        self.t = 1

    def evaluate_classifier(self, X, training=False):
        S_l = X
        for i in range(self.n_layers - 2):
            if self.BN:
                self.inputs_batch[i] = S_l

            self.hidden_layers_batch[i] = S_l = self.compute_hidden(S_l, i)

            if self.BN:
                self.hidden_layers_mod_batch[i] = S_l = self.batch_normalization(S_l, i, training)
                S_l = self.compute_scale_shift(S_l, i)

            S_l = np.maximum(S_l, np.zeros((self.layers[i+1], np.size(X, axis=1))))  # ReLu
        S_l = self.compute_hidden(S_l, self.n_layers - 2)
        P = softmax(S_l)
        #print(P)
        return P

    def batch_normalization(self, S_i, layer, training):
        if training:
            batch_mean_l, batch_variance_l = self.batch_prep(S_i)
            self.batch_means[layer] = batch_mean_l
            self.batch_variances[layer] = batch_variance_l
        else:
            batch_mean_l = self.batch_means_avg[layer]
            batch_variance_l = self.batch_variances_avg[layer]

        S_i = np.diagflat(1/(np.sqrt(batch_variance_l + eps))).dot(S_i - batch_mean_l.reshape((len(batch_mean_l), 1)))
        return S_i

    def batch_prep(self, S_i):
        batch_size = np.size(S_i, axis=1)
        batch_mean = np.sum(S_i, axis=1) / batch_size  # 13

        batch_variance = np.sum((S_i.transpose() - batch_mean).transpose() ** 2, axis=1) / batch_size # 14
        return batch_mean, batch_variance

    def compute_hidden(self, X, layer):
        S_l = self.weights[layer].dot(X) + self.biases[layer]
        return S_l

    def compute_scale_shift(self, S_l, layer):
        S_l = self.gammas[layer] * S_l + self.betas[layer]
        return S_l

    def compute_cost_and_loss(self, X, Y):
        norm_factor = 1 / np.size(X, axis=1)
        P = self.evaluate_classifier(X)
        sum_entropy = 0

        assert np.size(Y, axis=1) == np.size(P, axis=1)

        for i in range(np.size(Y, axis=1)):
            sum_entropy += self.cross_entropy(P[:, i], Y[:, i])

        weight_and_sum_score = 0
        for weight in self.weights:
            weight_and_sum_score += (np.sum(np.square(weight)))

        penalty_term = self.lamda * weight_and_sum_score
        cost = norm_factor * sum_entropy + penalty_term
        loss = norm_factor * sum_entropy
        return cost, loss

    def cross_entropy(self, p, y):
        # s: softmax network output
        # y: expected output - one-hot encoding
        return -np.log10(np.dot(np.array(y), p))

    def compute_accuracy(self, X, y):
        P = self.evaluate_classifier(X)
        print(P[:, 0])
        print(P[:, 0][0])
        if np.isnan(P[:, 0][0]):
            return 0
        correct_answers = 0
        assert np.size(P, axis=1) == np.size(X, axis=1)
        assert np.size(P, axis=1) == np.size(y)
        for i in range(np.size(P, axis=1)):
            highest_output_node = np.where(P[:, i] == np.amax(P[:, i]))
            if highest_output_node[0][0] == y[i]:
                correct_answers += 1

        return correct_answers / np.size(P, axis=1)

    def compute_gradients(self, X_batch, Y_batch, P_batch, batch_size, eta):
        # We backprop the gradient G through the net #
        G_batch = self.init_G_batch(Y_batch, P_batch)

        n_hidden_layers = self.n_layers - 2
        G_batch = self.update_network_params(G_batch, n_hidden_layers, eta, 0, batch_size)

        for l in range(n_hidden_layers-1):
            if self.BN:
                G_batch = self.update_batch_norm_params(G_batch, n_hidden_layers, eta, l+1, batch_size)
                #print(self.gammas)
                #print(self.betas)
            G_batch = self.update_network_params(G_batch, n_hidden_layers, eta, l+1, batch_size)

        if self.BN:
            self.update_batch_norm_params(G_batch, n_hidden_layers, eta, n_hidden_layers, batch_size)

        dloss_W1, dloss_b1 = self.get_weight_gradient(X_batch, G_batch, batch_size)
        gradient_W1 = dloss_W1 + 2 * self.lamda * self.weights[0]
        gradient_b1 = dloss_b1
        gradients = gradient_W1, gradient_b1
        self.update_weights(gradients, 0, eta)


    def init_G_batch(self, Y_batch, P_batch):
        return np.array(-(Y_batch - P_batch))

    def get_weight_gradient(self, layer_input_batch, G_batch, batch_size):
        dloss_W = G_batch.dot(layer_input_batch.transpose()) / batch_size
        dloss_b = G_batch.dot(np.ones((batch_size, 1))) / batch_size
        return dloss_W, dloss_b

    def propagate_G_batch(self, G_batch, weight, input):
        G_batch = weight.transpose().dot(G_batch)
        G_batch = G_batch * np.where(input > 0, input / input, input * 0)
        return G_batch

    def update_network_params(self, G_batch, n_hidden_layers, eta, l, batch_size):
        dloss_Wl, dloss_bl = self.get_weight_gradient(self.hidden_layers_batch[n_hidden_layers - l - 1], G_batch,
                                                      batch_size)
        G_batch = self.propagate_G_batch(G_batch, self.weights[n_hidden_layers - l],
                                         self.hidden_layers_batch[n_hidden_layers - l - 1])
        gradient_Wl = dloss_Wl + 2 * self.lamda * self.weights[n_hidden_layers - l]
        gradient_bl = dloss_bl
        gradients = gradient_Wl, gradient_bl
        self.update_weights(gradients, n_hidden_layers - l, eta)
        return G_batch

    def update_batch_norm_params(self, G_batch, n_hidden_layers, eta, l, batch_size):
        gradients = self.get_batch_gradient(G_batch, batch_size, n_hidden_layers - l)
        G_batch = G_batch*(self.gammas[n_hidden_layers-l].dot(np.ones((1, batch_size))))
        G_batch = self.BatchNormBackPass(G_batch, n_hidden_layers - l, batch_size)
        self.update_BM_params(gradients, n_hidden_layers - l, eta)
        return G_batch

    def get_batch_gradient(self, G_batch, batch_size, l):
        dloss_gammal = np.sum(G_batch * self.hidden_layers_mod_batch[l], axis=1)/batch_size
        dloss_betal = np.sum(G_batch)/batch_size
        return dloss_gammal, dloss_betal

    def BatchNormBackPass(self, G_batch, l, batch_size):
        sigma_1 = np.array((self.batch_variances[l] + eps)**(-0.5)) # 31
        sigma_2 = np.array((self.batch_variances[l] + eps)**(-1.5)) # 32
        sigma_1 = sigma_1.reshape((len(sigma_1), 1))
        sigma_2 = sigma_2.reshape((len(sigma_2), 1))
        G_1 = G_batch*(sigma_1.dot(np.ones((1, batch_size)))) # 33
        G_2 = G_batch*(sigma_2.dot(np.ones((1, batch_size)))) # 34
        D = self.hidden_layers_batch[l] - np.array(self.batch_means[l]).reshape((self.batch_means[l].size, 1)).dot(np.ones((1, batch_size))) # 35
        c = np.dot((G_2*D), np.ones(batch_size)) # 36

        dot_1 = np.dot(G_1, np.ones(batch_size))
        reshape_dot_1 = dot_1.reshape((len(dot_1), 1))
        term_1 = np.dot(reshape_dot_1, np.ones((1, batch_size)))/batch_size
        G_batch = G_1 - term_1 - D*np.dot(c.reshape((len(c), 1)), np.ones((1, batch_size)))/batch_size
        return G_batch

    def update_BM_params(self, gradients, weight, eta):
        gradient_gammal, gradient_betal = gradients

        gradient_gammal = np.reshape(gradient_gammal, (np.size(gradient_gammal), 1))
        gradient_betal = np.reshape(gradient_betal, (np.size(gradient_betal), 1))
        self.gammas[weight] = self.gammas[weight] - eta * gradient_gammal
        self.betas[weight] = self.betas[weight] - eta * gradient_betal

    def MiniBatchGD(self, train_data, val_data, GDparams):
        # TODO Random shuffle of train data after each epoch
        batch_size, epochs = GDparams

        # init information
        train_cost = []
        validation_cost = []
        train_loss = []
        validation_loss = []

        X, Y = train_data
        X_val, Y_val = val_data

        for i in range(epochs):
            X, Y = self.shuffle_data(X, Y)
            self.fit(X, Y, batch_size)

            t_cost, t_loss = self.compute_cost_and_loss(X, Y)
            val_cost, val_loss = self.compute_cost_and_loss(X_val, Y_val)

            train_cost.append(t_cost)
            validation_cost.append(val_cost)
            train_loss.append(t_loss)
            validation_loss.append(val_loss)

            if self.checkIfTrainingShouldStop():
                break

        cost = train_cost, validation_cost
        loss = train_loss, validation_loss

        return cost, loss

    def shuffle_data(self, X, Y):
        X, Y = sklearn.utils.shuffle(X.transpose(), Y.transpose())
        return X.transpose(), Y.transpose()

    def fit(self, X, Y, batchSize=-1):

        if (batchSize == -1):
            batchSize = 1

        for i in range(0, X.shape[1], batchSize):
            eta_t = self.updatedLearningRate()
            if self.checkIfTrainingShouldStop():
                break
            batchX = X[:, i:i + batchSize]
            batchY = Y[:, i:i + batchSize]
            batchP = self.evaluate_classifier(batchX, training=True)
            if self.BN:
                #self.final_prob_batch.append(batchP)
                if i is 0:
                    self.init_batch_avgs()
                self.update_batch_avgs()

            self.compute_gradients(batchX, batchY, batchP, batchSize, eta_t)
            self.t += 1


    def update_weights(self, gradients, weight, eta):
        gradient_Wl, gradient_bl = gradients

        gradient_bl = np.reshape(gradient_bl, (np.size(gradient_bl), 1))
        self.weights[weight] = self.weights[weight] - eta * gradient_Wl
        self.biases[weight] = self.biases[weight] - eta * gradient_bl

    def updatedLearningRate(self):
        l = np.floor(self.t / (2 * self.step_size))
        if 2 * l * self.step_size <= self.t < (2 * l + 1) * self.step_size:
            return self.evenIterationFunc(l, self.t)
        else:
            return self.unevenIterationFunc(l, self.t)

    def evenIterationFunc(self, l, t):
        return self.eta_min + (t - 2 * l * self.step_size) / self.step_size * (self.eta_max - self.eta_min)

    def unevenIterationFunc(self, l, t):
        return self.eta_max - (t - (2 * l + 1) * self.step_size) / self.step_size * (self.eta_max - self.eta_min)

    def checkIfTrainingShouldStop(self):
        return self.n_cycles * 2 == self.t / self.step_size

    def init_batch_avgs(self):
        self.batch_means_avg = self.batch_means
        self.batch_variances_avg = self.batch_variances

    def update_batch_avgs(self):
        for i in range(len(self.batch_means)):
            self.batch_means_avg[i] = self.alfa*self.batch_means_avg[i] + (1-self.alfa)*self.batch_means[i]
            self.batch_variances_avg[i] = self.alfa*self.batch_variances_avg[i] + (1-self.alfa)*self.batch_variances[i]

eps = sys.float_info.epsilon

BN = True
alfa = 0.9
#lamda = 0.012420743449115342
# 1 lamda = 0.010420743449115342
lamda = 0.003020743449115342
eta_min = 0.00001
eta_max = 0.1
batch_size = 100
epochs = 100
#step_size = 800
# step_size = 2 * np.floor(np.size(processed_training_data[0], axis=1) / batch_size)
step_size = 5 * 49000 / batch_size
n_cycles = 2
eta_params = eta_min, eta_max, step_size, n_cycles
GDparams = epochs, batch_size


def load_training_data():
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

    return training_data, validation_data, test_data


def pre_process_all_data(training_data, validation_data, test_data):
    proc_train = pre_process(training_data)
    proc_val = pre_process(validation_data)
    proc_test = pre_process(test_data)
    return proc_train, proc_val, proc_test


def generate_neural_net(proc_train):
    output_size = np.size(proc_train[1], axis=0)
    input_size = np.size(proc_train[0], axis=0)
    #layers = [input_size, 50, 50, output_size]
    layers = [input_size, 50, 50, output_size]

    return ANN_multilayer(layers, lamda, eta_params, BN=BN, alfa=alfa), layers


def setup_train_data(proc_train, proc_val):
    tdi = proc_train[0]
    tdl = proc_train[1]
    vdi = proc_val[0]
    vdl = proc_val[1]

    train_data = tdi, tdl
    val_data = vdi, vdl

    return train_data, val_data


def train_network(train_data, val_data):
    return neural_net.MiniBatchGD(train_data, val_data, GDparams)


def lamda_optimization(train_data, val_data, proc_train, proc_val, layers, eta_params, GDparams):
    iterations_c = 5
    best_lamda = 0
    high_val_acc = 0
    l_min = -5
    l_max = -1
    #lamdas_course = [1.10713074e-03, 2.43313844e-05, 7.50698134e-02, 2.67369941e-04, 1.12718786e-03]
    lamdas_course = np.random.uniform(10**(l_min), 10**(l_max), iterations_c)
    step_size = 2 * np.floor(np.size(proc_train[0], axis=1) / batch_size)

    print("COURSE SEARCH")
    for i in range(iterations_c):
        print("Iteration: ", i)
        neural_net = ANN_multilayer(layers, lamdas_course[i], eta_params, BN=BN, alfa=alfa)
        neural_net.MiniBatchGD(train_data, val_data, GDparams)
        val_accuracy = neural_net.compute_accuracy(proc_val[0], proc_val[2])
        if val_accuracy > high_val_acc:
            high_val_acc = val_accuracy
            best_lamda = i
            print("Best lamda: ", lamdas_course[best_lamda])
            print("Val_accuracy: ", high_val_acc)

    margins_of_search = 0.00001
    lamdas = lamdas_course[best_lamda] + np.random.uniform(-margins_of_search, margins_of_search, iterations_c)
    lamdas = np.concatenate((lamdas_course, lamdas))
    iterations = 5

    # print("\n")

    print("FINE SEARCH")
    for i in range(iterations_c, iterations_c + iterations):
        print("Iteration: ", i)
        neural_net = ANN_multilayer(layers, lamdas[i], eta_params, BN=BN, alfa=alfa)
        neural_net.MiniBatchGD(train_data, val_data, GDparams)
        val_accuracy = neural_net.compute_accuracy(proc_val[0], proc_val[2])
        if val_accuracy > high_val_acc:
            high_val_acc = val_accuracy
            best_lamda = i
            print("Best lamda: ", lamdas[best_lamda])
            print("Val_accuracy: ", high_val_acc)

    return lamdas[best_lamda]

def plot_cost_and_lost(cost, loss):
    train_cost, validation_cost = cost
    train_loss, validation_loss = loss
    plot_cost(np.array(train_cost), validation_cost)
    plot_total_loss(np.array(train_loss), validation_loss)

def print_net_performance(neural_net, proc_train, proc_val, proc_test):
    # train_cost, train_loss = neural_net.compute_cost_and_loss(processed_training_data[0], processed_training_data[1])'''

    '''print("-----------------------------")
    print("Final train loss: ", )
    print("Final validation loss: ",
          neural_net.compute_cost(processed_validation_data[0], processed_validation_data[1]))
    print("Final test loss: ", neural_net.compute_cost(processed_test_data[0], processed_test_data[1]))'''

    print("------------------------------")
    print("Final train accuracy: ", neural_net.compute_accuracy(proc_train[0], proc_train[2]))
    print("Final validation accuracy: ",
          neural_net.compute_accuracy(proc_val[0], proc_val[2]))
    print("Final test accuracy: ", neural_net.compute_accuracy(proc_test[0], proc_test[2]))

def grad_checks(proc_train, neural_net):
    grad_analytically, grad_numerically = ComputeGradients(proc_train[0], proc_train[1], neural_net,
                                                           batch_size)
    printOutGradients(grad_analytically, grad_numerically)

    eps = 0.001
    [grad_W, grad_b] = grad_analytically
    [grad_Wn, grad_bn] = grad_numerically

    print_gradient_check(grad_W, grad_Wn, grad_b, grad_bn, eps)

if __name__ == '__main__':
    train_data, val_data, test_data = load_training_data()
    proc_train, proc_val, proc_test = pre_process_all_data(train_data, val_data, test_data)
    train_input_output, val_input_output = setup_train_data(proc_train, proc_val)
    neural_net, layers = generate_neural_net(proc_train)
    #lamda = lamda_optimization(train_input_output, val_input_output, proc_train, proc_val, layers, eta_params, GDparams)
    cost, loss = train_network(train_input_output, val_input_output)
    plot_cost_and_lost(cost, loss)
    print_net_performance(neural_net, proc_train, proc_val, proc_test)
    #grad_checks(proc_train, neural_net)

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
