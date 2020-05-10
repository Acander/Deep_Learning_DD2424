import numpy as np
from Assignment3.Util import load_batch, pre_process, plot_total_loss, plot_cost
from Assignment3.Util import print_gradient_check, printOutGradients, ComputeGradients
from Assignment3.ANN_multilayer import ANN_multilayer

BN = True
alfa = 0.9
# lamda = 0.0010995835253050919
lamda = 0.005
eta_min = 0.00001
eta_max = 0.1
batch_size = 100
epochs = 100
#step_size = 800
# step_size = 2 * np.floor(np.size(processed_training_data[0], axis=1) / batch_size)
step_size = 5 * 45000 / batch_size
n_cycles = 2
eta_params = eta_min, eta_max, step_size, n_cycles
GDparams = epochs, batch_size


def load_training_data():
    [X_train_1, Y_train_1, y_train_1] = load_batch('data_batch_1')
    [X_train_2, Y_train_2, y_train_2] = load_batch('data_batch_2')
    [X_train_3, Y_train_3, y_train_3] = load_batch('data_batch_3')
    [X_train_4, Y_train_4, y_train_4] = load_batch('data_batch_4')
    [X_train_5, Y_train_5, y_train_5] = load_batch('data_batch_5')

    X_train_5, X_val = np.split(X_train_5, 2, axis=1)
    Y_train_5, Y_val = np.split(Y_train_5, 2, axis=1)
    y_train_5, y_val = np.split(y_train_5, 2)

    '''[X_train_5, X_val] = np.split(X_train_5, [9000], axis=1)
    [Y_train_5, Y_val] = np.split(Y_train_5, [9000], axis=1)
    [y_train_5, y_val] = np.split(y_train_5, [9000])'''

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


def lamda_optimization(train_data, val_data, layers, eta_params, GDparams):
    iterations_c = 5
    best_lamda = 0
    high_val_acc = 0.5208
    l_min = -5
    l_max = -1
    lamdas_course = [1.10713074e-03, 2.43313844e-05, 7.50698134e-02, 2.67369941e-04, 1.12718786e-03]
    step_size = 2 * np.floor(np.size(proc_train[0], axis=1) / batch_size)

    print("COURSE SEARCH")
    for i in range(iterations_c):
        print("Iteration: ", i)
        neural_net = ANN_multilayer(layers, lamdas_course[i], eta_params)
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
        neural_net = ANN_multilayer(layers, lamdas[i], eta_params)
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
    #lamda_optimization(proc_train, proc_val, layers, eta_params, GDparams)
    cost, loss = train_network(train_input_output, val_input_output)
    #plot_cost_and_lost(cost, loss)
    print_net_performance(neural_net, proc_train, proc_val, proc_test)
    #grad_checks(proc_train, neural_net)