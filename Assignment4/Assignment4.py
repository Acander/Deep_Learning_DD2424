import numpy as np
from array import array
from Assignment1.functions import softmax


class RNN:
    def __init__(self, k):
        self.m = 100
        self.k = k
        sigma = 0.01

        self.weights_U = np.random.normal(0, sigma, (self.m, self.k))
        self.weights_W = np.random.normal(0, sigma, (self.m, self.m))
        self.weights_V = np.random.normal(0, sigma, (self.k, self.m))

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
    return -np.log10(y-p)

def batch_forward_pass(X, rnn):
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
    return A, H, O, P

def compute_gradients(X, Y, h, rnn):
    tao = np.size(X, axis=1)
    A, H, O, P = batch_forward_pass(X, rnn)
    print("A ->", A)
    print("H ->", H)
    print("O ->", O)
    print("P ->", P)
    da = np.array((tao, rnn.m))
    dh = np.array((tao, rnn.m))
    loss = 0

    for i in range(tao):
        l = + cross_entropy(Y[:, i], P[:, i])

    dO = -(Y-P)
    dh[tao] = np.dot(dO[tao], rnn.weights_V)
    da[tao] = np.dot(dh[tao], np.diag(1-np.tanh(A[:, tao])))

    for i in range(tao, 0, -1):
        dh[i] = np.dot(dO[i], rnn.weights_V) + np.dot(da[i+1], rnn.weights_W)
        da[i] = np.dot(dh[i], np.diag(1-np.tanh(A[i]))**2)

    dc = np.sum(dO)

    dW = 0
    dV = 0
    dW = dW + np.dot(da[0], H)
    for i in range(1, tao):
        dW = dW + np.dot(da[i], H[i-1])
        dV = dV + np.dot(dO[i-1], H[i - 1])

    dV = dV + np.dot(dO[tao], H[tao])
    dW = dW + np.dot(da[0], h)

    db = np.sum(da)
    dU = np.dot(da, X)

    h = H[tao]

    return Gradients(dU, dW, dV, db, dc)

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

def create_train_dataset(book_data, seq_len, char_table):
    X_chars = book_data[0:seq_len]
    Y_chars = book_data[1:seq_len+1]

    nr_chars = len(char_table)

    X = []
    Y = []
    for i in range(seq_len):
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
    X, Y = create_train_dataset(book, 10, char_set)
    #print("X-data", X)
    #print("Y-data", Y)
    rnn = RNN(len(char_set))
    h = np.zeros((rnn.m, 1))
    compute_gradients(X, Y, h, rnn)


if __name__ == '__main__':
    #run()

    # TESTING
    #######################################
    #sequence_testing()
    #dataset_testing()
    test_back_prop()