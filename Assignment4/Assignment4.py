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


def synthesize_sequence(rnn, h0, x0, n, chars):
    ht = h0
    xt = x0
    final_sequence = []
    for _ in range(n):
        at = rnn.weights_W.dot(ht) + rnn.weights_U.dot(xt) + rnn.bias_b # at = W ht−1 + Uxt + b (1)
        ht = np.tanh(at) # ht = tanh(at) (2)
        ot = rnn.weights_V.dot(ht) + rnn.bias_c # ot = V ht + c (3)
        pt = softmax(ot) # pt = SoftMax(ot)
        #char_index = np.argmax(pt)
        char_index = np.random.choice(a=np.arange(0, rnn.k), size=1, p=pt)[0]
        final_sequence.append(chars[char_index])
        xt = create_one_hot_encoding(char_index, rnn.k)

    return ''.join(final_sequence)

def load_book():
    book = open("Datasets/goblet_book.txt", 'r').read()
    return book


def set_chars(book_data):
    return list(set(book_data))


def create_one_hot_encoding(index, nr_chars):
    array = np.zeros(nr_chars)
    array[index] = 1
    return array

def create_train_dataset(book_data, seq_len):
    X_chars = book_data[0:seq_len]
    Y_chars = book_data[1:seq_len+1]
    return X_chars, Y_chars

def run():
    book_data = load_book()
    book_chars = set_chars(book_data)
    print(create_one_hot_encoding(1, len(book_data)))

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
    char_set = set_chars(book)
    print(char_set)
    rnn = RNN(len(char_set))
    x0 = np.zeros(rnn.k)
    x0[0] = 1
    print(char_set[0])
    sequence = synthesize_sequence(rnn, np.zeros(rnn.m), x0, 10, char_set)
    print(sequence)

def dataset_testing():
    book = load_book()
    create_train_dataset(book, 10)

if __name__ == '__main__':
    #run()

    # TESTING
    #######################################
    #sequence_testing()
    dataset_testing()