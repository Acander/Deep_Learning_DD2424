import numpy as np
from array import array
from Assignment1.functions import softmax

TAU = 10

class RNN:
    def __init__(self, k):
        self.m = 100
        self.k = k
        sigma = 0.01

        self.weights_U = np.random.normal(0, sigma, (self.m, self.k)) #Input
        self.weights_W = np.random.normal(0, sigma, (self.m, self.m)) #Hidden to Hidden
        self.weights_V = np.random.normal(0, sigma, (self.k, self.m)) #Output

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
    at = rnn.weights_W.dot(ht) + rnn.weights_U.dot(xt) + rnn.bias_b  # at = W ht−1 + Uxt + b (1)
    ht = np.tanh(at)  # ht = tanh(at) (2)
    ot = rnn.weights_V.dot(ht) + rnn.bias_c  # ot = V ht + c (3)
    pt = softmax(ot)  # pt = SoftMax(ot)
    # char_index = np.argmax(pt)

    char_index = np.random.choice(a=np.arange(0, rnn.k), size=1, p=pt)[0]
    fp_output = at, ht, ot, pt
    return char_index, fp_output


def cross_entropy(y, p):
    return -np.log10(np.reshape(y, (1, len(y))).dot(p)[0])


def batch_forward_pass(X, Y, rnn):
    final_sequence = []
    A = []
    H = []
    O = []
    P = []
    h0 = np.zeros(rnn.m)
    for i in range(np.size(X, axis=0)):
        char_index, fp_output = forward_pass(rnn, h0, X[i])
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

    return [A, H, O, P], loss

def compute_gradients(X, Y, h, rnn):
    tao = np.size(X, axis=0)
    [A, H, O, P], loss = batch_forward_pass(X, Y, rnn)
    da = np.zeros((tao, rnn.m))
    dh = np.zeros((tao, rnn.m))

    dO = np.array(-(Y-P))

    dh[tao-1] = np.dot(dO[tao-1], rnn.weights_V)

    da[tao-1] = np.dot(dh[tao-1], np.diag(1-np.tanh(A[tao-1])))

    for i in range(tao-2, -1, -1):
        dh[i] = np.dot(dO[i], rnn.weights_V) + np.dot(da[i+1], rnn.weights_W)
        da[i] = np.dot(dh[i], np.diag(1-np.tanh(A[i]))**2)

    dc = np.sum(dO, axis=0)

    dW = 0
    dV = 0

    dW += np.dot(np.reshape(da[0], (len(da[0]), 1)), h.transpose())

    for i in range(1, tao):
        dW += np.dot(np.reshape(da[0], (len(da[0]), 1)), np.reshape(H[i - 1], (1, len(H[i - 1]))))
        dV += np.dot(np.reshape(dO[i-1], (len(dO[i-1]), 1)), np.reshape(H[i - 1], (1, len(H[i - 1])))) #TODO CHECK IF BOTH SHOULD BE TRANSPOSED

    dV += np.dot(np.reshape(dO[tao-1], (len(dO[tao-1]), 1)), np.reshape(H[tao - 1], (1, len(H[tao - 1]))))

    db = np.sum(da, axis=0)
    dU = np.dot(da.transpose(), X)

    #h = H[tao-1]
    gradients = Gradients(dU, dW, dV, db, dc)
    return gradients, loss

#LOAD BOOK DATA AND CREATE LIST OF UNIQUE CHARS
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

######################################################################

def create_train_dataset(book_data, tau, char_table):
    X_chars = book_data[0:tau]
    Y_chars = book_data[1:tau+1]

    nr_chars = len(char_table)

    X = []
    Y = []
    for i in range(tau):
        X.append(create_one_hot_encoding(char_table.index(X_chars[i]), nr_chars))
        Y.append(create_one_hot_encoding(char_table.index(Y_chars[i]), nr_chars))
    return X, Y

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
    X, Y = create_train_dataset(book, TAU, char_set)
    #print("X-data", X)
    #print("Y-data", Y)
    rnn = RNN(len(char_set))
    h = np.zeros((rnn.m, 1))
    train_data = X, Y
    grad_check(rnn, train_data, h)

def numericalGradients(X, Y, rnn, h=1e-4):
    numV = np.zeros(rnn.weights_V.shape)
    for i in range(rnn.weights_V.shape[0]):
        for j in range(rnn.weights_V.shape[1]):
            orgVal = rnn.weights_V[i, j]

            rnn.weights_V[i, j] = orgVal - h
            _, l1 = batch_forward_pass(X, Y, rnn)
            rnn.weights_V[i, j] = orgVal + h
            _, l2 = batch_forward_pass(X, Y, rnn)
            numV[i, j] = (l2 - l1) / (2 * h)

            rnn.weights_V[i, j] = orgVal

    # W
    numW = np.zeros(rnn.weights_W.shape)
    for i in range(rnn.weights_W.shape[0]):
        for j in range(rnn.weights_W.shape[1]):
            orgVal = rnn.weights_W[i, j]

            rnn.weights_W[i, j] = orgVal - h
            _, l1 = batch_forward_pass(X, Y, rnn)
            rnn.weights_W[i, j] = orgVal + h
            _, l2 = batch_forward_pass(X, Y, rnn)
            numW[i, j] = (l2 - l1) / (2 * h)

            rnn.weights_W[i, j] = orgVal

    # U
    numU = np.zeros(rnn.weights_U.shape)
    for i in range(rnn.weights_U.shape[0]):
        for j in range(rnn.weights_U.shape[1]):
            orgVal = rnn.weights_U[i, j]

            rnn.weights_U[i, j] = orgVal - h
            _, l1 = batch_forward_pass(X, Y, rnn)
            rnn.weights_U[i, j] = orgVal + h
            _, l2 = batch_forward_pass(X, Y, rnn)
            numU[i, j] = (l2 - l1) / (2 * h)

            rnn.weights_U[i, j] = orgVal

    # b
    numB = np.zeros(rnn.bias_b.shape)
    for i in range(numB.shape[0]):
        orgVal = rnn.bias_b[i]

        rnn.bias_b[i] = orgVal - h
        _, l1 = batch_forward_pass(X, Y, rnn)
        rnn.bias_b[i] = orgVal + h
        _, l2 = batch_forward_pass(X, Y, rnn)
        numB[i] = (l2 - l1) / (2 * h)

        rnn.bias_b[i] = orgVal

    # c
    numC = np.zeros(rnn.bias_c.shape)
    for i in range(numC.shape[0]):
        orgVal = rnn.bias_c[i]

        rnn.bias_c[i] = orgVal - h
        _, l1 = batch_forward_pass(X, Y, rnn)
        rnn.bias_c[i] = orgVal + h
        _, l2 = batch_forward_pass(X, Y, rnn)
        numC[i] = (l2 - l1) / (2 * h)

        rnn.bias_c[i] = orgVal

    # print("Arrays are equal V:", np.array_equal(VBackup, rnn.V))
    # print("Arrays are equal W:", np.array_equal(WBackup, rnn.W))
    return numV, numW, numU, numB, numC

def grad_check(rnn, train_data, h):
    X, Y = train_data

    gradients, _ = compute_gradients(X, Y, h, rnn)
    num_dV, num_dW, num_dU, num_db, num_dc = numericalGradients(X, Y, rnn)
    dV = gradients.g_V
    dW = gradients.g_W
    dU = gradients.g_U
    db = gradients.g_b
    dc = gradients.g_c

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

if __name__ == '__main__':
    #run()

    # TESTING
    #######################################
    #sequence_testing()
    #dataset_testing()
    test_back_prop()