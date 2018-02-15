import numpy as np
from utils import *
import os

def layer_sizes(X, Y):
    '''
    :param X: input dataset 
    :param Y: labels of shape
    :return: 
    n_x --- the size of the input layer
    n_h --- the size of the hidden layer (one hidden layer)
    n_y --- the size of the output layer
    '''
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    '''
    :param n_x: 
    :param n_h: 
    :param n_y: 
    :return: 
    params --- python dictionary containing your parameters:
        W1 --- weight matrix of shape (n_h, n_x)
        b1 --- bias vector of shape (n_h, 1)
        W2 --- weight matrix of shape (n_y, n_h)
        b2 --- bias vector of shape (n_y, 1)
    '''
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}
    return parameters

def forward_propagation(X, parameters):
    '''
    :param X: input data of shpe(n_x, m)
    :param parameters: input dictionary
    :return: 
    A2 --- sigmoid output
    cache --- a dictionary contains "Z1\A1\Z2\A2"
    '''
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp( -Z2))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):
    '''
    :param A2: The sigmoid output 
    :param Y: label vector
    :param parameters: parameters dictionary
    :return: 
    cost --- cost
    '''
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m

    return cost

def back_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    '''
    Updates parameters using the gradient descent update rule.
    :param parameters: 
    :param grads: 
    :param learning_rate: 
    :return: 
    parameters --- update parameters
    '''
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = True):
    '''
    combine all funcs
    :param X: data 
    :param Y: labels
    :param n_h: size of hidden layer
    :param num_iterations: 
    :param print_cost: 
    :return: 
    parameters --- learnt by the model
    '''
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = back_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predict...
    :param parameters: dictionary parameters
    :param X: test data
    :return: 
    predictions --- vector of predictions
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)    # 0.5 is the edge

    return predictions

def load_dataset(path):
    files = os.listdir(path)
    res = None
    Y = []
    first = True
    for file in files:
        if(file.endswith(".jpg") == False):
            continue
        array = image2matrix(path + '/' + file, display=False)
        log(array)
        if(first == True):
            first = False
            res = array
            Y.append(file[0])
            continue
        res = np.c_[res, array]
        Y.append(file[0])

    X = res
    return X, Y

X , Y = load_dataset("D:/neu_code")
parameters = nn_model(X, Y, 4, 1000)
print(parameters)
predict()

