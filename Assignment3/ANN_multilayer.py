import numpy as np
import sys
import sklearn
from Assignment1.functions import softmax

eps = sys.float_info.epsilon

class ANN_multilayer:

    def __init__(self, layers, lamda, eta_params, BN=False, alfa=0):
        '''The 'layers' parameter is an array of integers, each representing the size of a hidden layer.'''

        mu, sigma = 0, 0.01
        self.n_layers = len(layers)  # Including input and output layers
        print("n_layers: ", self.n_layers)
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
        '''self.batch_means = np.array(self.n_layers)
        self.batch_variances = np.array(self.n_layers)
        self.batch_means_avg = np.array(self.n_layers)
        self.batch_variances_avg = np.array(self.n_layers)'''
        self.batch_means = []
        self.batch_variances = []
        self.batch_means_avg = []
        self.batch_variances_avg = []
        self.gammas = []
        self.betas = []

        for i in range(self.n_layers - 1):
            self.weights.append(np.random.normal(mu, sigma, (layers[i + 1], layers[i])))
            self.biases.append(np.zeros((layers[i + 1], 1)))
            if BN:
                self.gammas.append(np.random.normal(mu, sigma, (layers[i + 1], 1)))
                self.betas.append(np.zeros((layers[i + 1], 1)))
                self.batch_means.append(np.array(layers[i+1]))
                self.batch_variances.append(np.array(layers[i+1]))
                self.batch_means_avg.append(np.array(layers[i+1]))
                self.batch_variances_avg.append(np.array(layers[i+1]))

        for i in range(self.n_layers-2):
            self.hidden_layers_batch.append(np.matrix((layers[i + 1], 1)))
            if BN:
                self.inputs_batch.append(np.matrix((layers[i + 1], 1)))
                self.hidden_layers_mod_batch.append(np.matrix((layers[i + 1], 1)))
                # self.networks_final_prob_batch.append(np.matrix((layers[i + 1], 1)))


        # Parameters related to cyclic learning rate:
        self.eta_min, self.eta_max, self.step_size, self.n_cycles = eta_params
        self.t = 1

    def evaluate_classifier(self, X, training=False):
        S_l = X
        for i in range(self.n_layers - 2):
            hidden_index = i + 1
            if self.BN:
                self.inputs_batch[i] = S_l

            self.hidden_layers_batch[i] = S_l = self.compute_hidden(S_l, hidden_index)

            if self.BN:
                self.hidden_layers_mod_batch[i] = S_l = self.batch_normalization(S_l, hidden_index, training)
                S_l = self.compute_scale_shift(S_l, hidden_index)

            S_l = np.maximum(S_l, np.zeros((self.layers[hidden_index], np.size(X, axis=1))))  # ReLu
        S_l = self.compute_hidden(S_l, self.n_layers - 1)
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
        #print(np.linalg.inv(np.sqrt(np.diagflat(batch_variance_l + eps))))
        #print(S_i - batch_mean_l.reshape((len(batch_mean_l), 1)))
        S_i = (np.linalg.inv(np.sqrt(np.diagflat(batch_variance_l + eps)))).dot(S_i - batch_mean_l.reshape((len(batch_mean_l), 1)))
        return S_i

    def batch_prep(self, S_i):
        batch_size = np.size(S_i, axis=1)
        batch_mean = np.sum(S_i, axis=1) / batch_size  # 13

        batch_variance = np.sum(np.subtract(S_i.transpose(), batch_mean) ** 2, axis=0) / batch_size # 14
        return batch_mean, batch_variance

    def compute_hidden(self, X, layer):
        S_l = self.weights[layer - 1].dot(X) + self.biases[layer - 1]
        return S_l

    def compute_scale_shift(self, S_l, layer):
        S_l = np.diag(self.gammas[layer - 1])* S_l + self.betas[layer - 1]
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
            G_batch = self.update_network_params(G_batch, n_hidden_layers, eta, l+1, batch_size)


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
        gradients = self.get_batch_gradient(G_batch, batch_size, l)
        G_batch = G_batch*(self.gammas[l].dot(np.ones((1, batch_size))))
        G_batch = self.BatchNormBackPass(G_batch, l, batch_size)
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