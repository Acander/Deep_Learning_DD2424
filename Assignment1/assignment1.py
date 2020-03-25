from Assignment1.functions import LoadBatch
import numpy as np


# n is the number of images
# d is the dimensionality of each image (3072=32x32x3)
def load_batch(filename):
    dict = LoadBatch(filename)
    X = np.array(dict[b'data']) # matrix, dim: image pixel data, # of images * dim
    # Y = matrix, diM: # of images * # of labels, one-hot representation
    y = np.array(dict[b'labels'])  # vector, contains labels, dim: # of images

    Y = make_one_hot_encoding(len(X), y)

    return [X.astype(float), Y.astype(float), y.astype(float)]


def make_one_hot_encoding(batch_size, indices):
    Y = np.zeros((batch_size, 10))
    for i in range(batch_size):
        hot = indices[i]
        Y[i][hot] = 1

    return Y.transpose()

#TODO: Check if that mean becomes approximatly zero is okay
#Standard Score? (X-mean)/std
def pre_process(training_data):
    [X, Y, y] = training_data
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)

    X = X - np.tile(mean_X, (np.size(X, axis=0), 1))
    X = X / np.tile(std_X, (np.size(X, axis=0), 1))
    return [X, Y, y]


if __name__ == '__main__':
    training_data = load_batch('data_batch_1')
    validation_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')

    processed_training_data = pre_process(training_data)
    processed_validation_data = pre_process(validation_data)
    processed_test_data = pre_process(test_data)

