from Assignment1.functions import LoadBatch
import numpy as np


# n is the number of images
# d is the dimensionality of each image (3072=32x32x3)
def load_batch(filename):
    dict = LoadBatch(filename)
    X = dict[b'data']  # matrix, dim: image pixel data, # of images * dim
    # Y = matrix, diM: # of images * # of labels, one-hot representation
    y = dict[b'labels']  # vector, contains labels, dim: # of images

    Y = make_one_hot_encoding(len(X), y)

    return [X, Y, y]


def make_one_hot_encoding(batch_size, indices):
    Y = np.zeros((batch_size, 10))
    for i in range(batch_size):
        hot = indices[i]
        Y[i][hot] = 1

    return Y


if __name__ == '__main__':
    [X, Y, y] = load_batch('data_batch_1')

