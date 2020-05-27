import numpy as np
import matplotlib.pyplot as plt
import pickle
from array import array
from Assignment1.functions import softmax

TAO = 25
ETA = 0.1
EPS = 1e-10
EPOCHS = 1
SYNTH_LEN = 200
SYNTH_STEP = 500
LOSS_PRINT_STEP = 100

RNN_FILE_NAME = "TRAINED_RNN.obj"

class RNN:
    def __init__(self, k):
        self.m = 5
        self.k = k
        sigma = 0.01

        self.weights_U = np.random.normal(0, sigma, (self.m, self.k))  # Input
        self.weights_W = np.random.normal(0, sigma, (self.m, self.m))  # Hidden to Hidden
        self.weights_V = np.random.normal(0, sigma, (self.k, self.m))  # Output

        self.bias_b = np.zeros(self.m)
        self.bias_c = np.zeros(self.k)


class Gradients:
    def __init__(self, g_U, g_W, g_V, g_b, g_c):
        self.g_U = g_U
        self.g_W = g_W
        self.g_V = g_V
        self.g_b = g_b
        self.g_c = g_c
        self.n_fields = 5

    def clip_gradients(self):
        self.g_U = self.clip(self.g_U)
        self.g_W = self.clip(self.g_W)
        self.g_V = self.clip(self.g_V)
        self.g_b = self.clip(self.g_b)
        self.g_c = self.clip(self.g_c)

    def clip(self, grad):
        return np.maximum(np.minimum(grad, 5), -5)


def saveRNN(rnn):
    fileHandler = open(RNN_FILE_NAME, 'wb')
    pickle.dump(rnn, fileHandler)


def loadRNN():
    fileHandler = open(RNN_FILE_NAME, 'rb')
    return pickle.load(fileHandler)


def synthesize_sequence(rnn, h0, x0, n, chars):
    ht = h0
    xt = x0
    final_sequence = []
    for _ in range(n):
        char_index, _ = forward_pass(rnn, ht, xt)
        final_sequence.append(chars[char_index])
        xt = create_one_hot_encoding(char_index, rnn.k)

    return ''.join(final_sequence)


def forward_pass(rnn, ht, xt):
    at = rnn.weights_W @ ht + rnn.weights_U @ xt + rnn.bias_b  # at = W ht−1 + Uxt + b (1)
    '''print("W: ", rnn.weights_W)
    print("U: ", rnn.weights_U)
    print("b: ", rnn.bias_b)'''
    ht = np.tanh(at)  # ht = tanh(at) (2)
    ot = rnn.weights_V @ ht + rnn.bias_c  # ot = V ht + c (3)
    pt = softmax(ot)  # pt = SoftMax(ot)
    #print("--------------------------------------------------------")
    '''print(at)
    print(ht)
    print(ot)
    print(pt)'''
    char_index = np.random.choice(a=np.arange(0, rnn.k), size=1, p=pt)[0]
    fp_output = at, ht, ot, pt
    return char_index, fp_output


def cross_entropy(y, p):
    return -np.log(np.reshape(y, (1, len(y))).dot(p)[0])


def batch_forward_pass(X, Y, ht, rnn):
    final_sequence = []
    A = []
    H = []
    O = []
    P = []
    for i in range(TAO):
        char_index, fp_output = forward_pass(rnn, ht, X[i])
        final_sequence.append(char_index)
        at, ht, ot, pt = fp_output
        A.append(at)
        H.append(ht)
        O.append(ot)
        P.append(pt)

    A = np.array(A)
    H = np.array(H)
    O = np.array(O)
    P = np.array(P)

    loss = 0
    for i in range(np.size(X, axis=0)):
        loss += cross_entropy(Y[i], P[i])

    return [A, H, O, P], loss, ht


def compute_gradients(X, Y, h, rnn, smooth_loss):
    tao = TAO
    [A, H, O, P], loss, _ = batch_forward_pass(X, Y, h, rnn)
    da = np.zeros((tao, rnn.m))
    dh = np.zeros((tao, rnn.m))
    dO = np.array(-(Y - P))

    dh[tao - 1] = dO[tao - 1] @ rnn.weights_V

    da[tao - 1] = dh[tao - 1] @ np.diag(1 - np.square(np.tanh(A[tao - 1])))

    for i in range(tao - 2, -1, -1):
        dh[i] = dO[i] @ rnn.weights_V + da[i + 1] @ rnn.weights_W
        da[i] = dh[i] @ np.diag(1 - np.square(np.tanh(A[i])))

    dc = np.sum(dO, axis=0)

    dW = 0
    dV = 0

    dW += np.reshape(da[0], (len(da[0]), 1)) @ np.reshape(h, (1, len(da[0])))

    for i in range(1, tao):
        dW += np.reshape(da[i], (len(da[i]), 1)) @ np.reshape(H[i - 1], (1, len(H[i - 1])))
        dV += np.reshape(dO[i - 1], (len(dO[i - 1]), 1)) @ np.reshape(H[i - 1], (1, len(H[i - 1])))

    dV += np.reshape(dO[tao - 1], (len(dO[tao - 1]), 1)) @ np.reshape(H[tao - 1], (1, len(H[tao - 1])))

    db = np.sum(da, axis=0)
    dU = da.transpose() @ X

    h = H[tao - 1]

    gradients = Gradients(dU, dW, dV, db, dc)
    gradients.clip_gradients()

    if smooth_loss is None:
        smooth_loss = loss
    else:
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

    return gradients, smooth_loss, h


def train_RNN():
    book = load_book()
    char_set = char_lookup_table(book)
    smooth_loss = None

    rnn = RNN(len(char_set))
    mU, mW, mV, mb, mc = 0, 0, 0, 0, 0
    M = mU, mW, mV, mb, mc

    smooth_loss_data = []

    print("Starting training")
    print("Nr of Epochs: ", EPOCHS)
    print(int(len(book)/TAO))
    for epoch in range(EPOCHS):
        hprev = np.zeros(rnn.m)
        iter_step = 0
        for e in range(0, len(book), TAO):

            if e > len(book) - 1 - TAO:
                break

            X, Y = create_train_dataset(book, TAO, char_set, e)

            gradients, smooth_loss, h = compute_gradients(X, Y, hprev, rnn, smooth_loss)
            M = update_weights(rnn, gradients, M)

            if iter_step % LOSS_PRINT_STEP == 0:
                print("Epoch:", epoch, "Iter =", iter_step, "Smooth_loss: ", smooth_loss, "Percentage of epoch done:", e/len(book), "%")
                smooth_loss_data.append(smooth_loss)

            if iter_step % SYNTH_STEP == 0:
                print(synthesize_sequence(rnn, hprev, X[0], SYNTH_LEN, char_set))

            hprev = h
            iter_step += 1

        plt.plot(np.arange(np.size(smooth_loss_data)), np.array(smooth_loss_data), color='blue', label='Smooth Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Smooth Loss')
        plt.show()

    saveRNN(rnn)

'''def update_weights(rnn, gradients, M):
    mU, mW, mV, mb, mc = M
    mU, rnn.weights_U = ada_grad(mU, rnn.weights_U, gradients.g_U)
    mW, rnn.weights_W = ada_grad(mW, rnn.weights_W, gradients.g_W)
    mV, rnn.weights_V = ada_grad(mV, rnn.weights_V, gradients.g_V)
    mb, rnn.bias_b = ada_grad(mb, rnn.bias_b, gradients.g_b)
    mc, rnn.bias_c = ada_grad(mc, rnn.bias_c, gradients.g_c)
    M = mU, mW, mV, mb, mc
    return M, rnn


def ada_grad(m, theta, grad):
    m = m + np.square(grad)
    theta = theta - (ETA/np.sqrt(m + EPS)) * theta
    return m, theta'''


def update_weights(rnn, gradients, M):
    mU, mW, mV, mb, mc = M
    mU = mU + np.square(gradients.g_U)
    mW = mW + np.square(gradients.g_W)
    mV = mV + np.square(gradients.g_V)
    mb = mb + np.square(gradients.g_b)
    mc = mc + np.square(gradients.g_c)
    rnn.weights_U = rnn.weights_U - (ETA / np.sqrt(mU + EPS)) * gradients.g_U
    rnn.weights_W = rnn.weights_W - (ETA / np.sqrt(mW + EPS)) * gradients.g_W
    rnn.weights_V = rnn.weights_V - (ETA / np.sqrt(mV + EPS)) * gradients.g_V
    rnn.bias_b = rnn.bias_b - (ETA / np.sqrt(mb + EPS)) * gradients.g_b
    rnn.bias_c = rnn.bias_c - (ETA / np.sqrt(mc + EPS)) * gradients.g_c
    M = mU, mW, mV, mb, mc
    return M

'''def update_weights_GD(rnn, gradients, M):
    rnn.weights_U = rnn.weights_U - ETA * rnn.gra_U
    rnn.weights_W = rnn.weights_W - ETA * rnn.weights_W
    rnn.weights_V = rnn.weights_V - ETA * rnn.weights_V
    rnn.bias_b = rnn.bias_b - ETA * rnn.bias_b
    rnn.bias_c = rnn.bias_c - ETA * rnn.bias_c'''

# LOAD BOOK DATA AND CREATE LIST OF UNIQUE CHARS
#####################################################################
def load_book():
    book = open("Datasets/goblet_book.txt", 'r').read()
    return book


def char_lookup_table(book_data):
    return list(set(book_data))


def create_one_hot_encoding(index, nr_chars):
    array = np.zeros(nr_chars)
    array[index] = 1
    return array


def create_train_dataset(book_data, tao, char_table, e):
    X_chars = book_data[e:e + tao]
    Y_chars = book_data[e + 1:e + tao + 1]

    nr_chars = len(char_table)

    X = []
    Y = []
    for i in range(tao):
        X.append(create_one_hot_encoding(char_table.index(X_chars[i]), nr_chars))
        Y.append(create_one_hot_encoding(char_table.index(Y_chars[i]), nr_chars))
    return X, Y


######################################################################

def run():
    book_data = load_book()
    char_table = char_lookup_table(book_data)
    print(create_one_hot_encoding(1, len(char_table)))


def sequence_testing():
    '''final_sequence = []
    final_sequence.append('a')
    print(final_sequence)
    final_sequence.append(' ')
    print(final_sequence)
    final_sequence.append('r')
    print(final_sequence)
    print(''.join(final_sequence))'''

    book = load_book()
    char_set = char_lookup_table(book)
    print(char_set)
    rnn = RNN(len(char_set))
    x0 = np.zeros(rnn.k)
    x0[0] = 1
    print(char_set[0])
    sequence = synthesize_sequence(rnn, np.zeros(rnn.m), x0, 10, char_set)
    print(sequence)


def test_back_prop():
    book = load_book()
    char_set = char_lookup_table(book)
    X, Y = create_train_dataset(book, TAO, char_set, 0)
    rnn = RNN(len(char_set))
    h = np.zeros(rnn.m)
    train_data = X, Y
    grad_check(rnn, train_data, h)


def numericalGradients(X, Y, rnn, h=1e-4):
    numV = np.zeros(rnn.weights_V.shape)
    # h0 = np.zeros(rnn.m)
    for i in range(rnn.weights_V.shape[0]):
        for j in range(rnn.weights_V.shape[1]):
            orgVal = rnn.weights_V[i, j]

            rnn.weights_V[i, j] = orgVal - h
            _, l1, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
            rnn.weights_V[i, j] = orgVal + h
            _, l2, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
            numV[i, j] = (l2 - l1) / (2 * h)

            rnn.weights_V[i, j] = orgVal

    # W
    numW = np.zeros(rnn.weights_W.shape)
    # h0 = np.zeros(rnn.m)
    for i in range(rnn.weights_W.shape[0]):
        for j in range(rnn.weights_W.shape[1]):
            orgVal = rnn.weights_W[i, j]

            rnn.weights_W[i, j] = orgVal - h
            _, l1, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
            rnn.weights_W[i, j] = orgVal + h
            _, l2, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
            numW[i, j] = (l2 - l1) / (2 * h)

            rnn.weights_W[i, j] = orgVal

    # U
    numU = np.zeros(rnn.weights_U.shape)
    # h0 = np.zeros(rnn.m)
    for i in range(rnn.weights_U.shape[0]):
        for j in range(rnn.weights_U.shape[1]):
            orgVal = rnn.weights_U[i, j]

            rnn.weights_U[i, j] = orgVal - h
            _, l1, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
            rnn.weights_U[i, j] = orgVal + h
            _, l2, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
            numU[i, j] = (l2 - l1) / (2 * h)

            rnn.weights_U[i, j] = orgVal

    # b
    numB = np.zeros(rnn.bias_b.shape)
    # h0 = np.zeros(rnn.m)
    for i in range(numB.shape[0]):
        orgVal = rnn.bias_b[i]

        rnn.bias_b[i] = orgVal - h
        _, l1, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
        rnn.bias_b[i] = orgVal + h
        _, l2, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
        numB[i] = (l2 - l1) / (2 * h)

        rnn.bias_b[i] = orgVal

    # c
    numC = np.zeros(rnn.bias_c.shape)
    # h0 = np.zeros(rnn.m)
    for i in range(numC.shape[0]):
        orgVal = rnn.bias_c[i]

        rnn.bias_c[i] = orgVal - h
        _, l1, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
        rnn.bias_c[i] = orgVal + h
        _, l2, _ = batch_forward_pass(X, Y, np.zeros(rnn.m), rnn)
        numC[i] = (l2 - l1) / (2 * h)

        rnn.bias_c[i] = orgVal

    # print("Arrays are equal V:", np.array_equal(VBackup, rnn.V))
    # print("Arrays are equal W:", np.array_equal(WBackup, rnn.W))
    return numV, numW, numU, numB, numC


def grad_check(rnn, train_data, h):
    X, Y = train_data

    gradients, _, _ = compute_gradients(X, Y, h, rnn, 0)
    num_dV, num_dW, num_dU, num_db, num_dc = numericalGradients(X, Y, rnn)
    dV = gradients.g_V
    dW = gradients.g_W
    dU = gradients.g_U
    db = gradients.g_b
    dc = gradients.g_c

    '''print("Analytically vs Nummerically ->")
    print("V:", dV, num_dV)
    print("W:", dW, num_dW)
    print("U:", dU, num_dU)
    print("b:", db, num_db)
    print("c:", dc, num_dc)'''

    maxRelError(num_dV, dV, weight_name="V")
    maxRelError(num_dW, dW, weight_name="W")
    maxRelError(num_dU, dU, weight_name="U")
    maxRelError(num_db, db, weight_name="b")
    maxRelError(num_dc, dc, weight_name="c")


def maxRelError(grad_n, grad_a, weight_name):
    absErr = np.sum(np.abs(grad_a - grad_n))
    absErrSum = np.sum(np.abs(grad_a)) + np.sum(np.abs(grad_n))
    relErr = absErr / max(1e-4, absErrSum)
    print(str(weight_name) + ":", relErr)


def test():
    array = np.array([0, 1, 0, 1, 0, 1])
    print(array.reshape((len(array), 1)))



if __name__ == '__main__':
    # run()
    train_RNN()

    # TESTING
    #######################################
    # sequence_testing()
    # dataset_testing()
    #test_back_prop()
    # test()
