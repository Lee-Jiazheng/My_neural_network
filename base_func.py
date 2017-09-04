import numpy as np

def sigmoid(z):
    '''
    Compute the sigmoid of z
    :return: 
    s --- sigmoid(z)
    '''
    s = 1 / (1 + np.exp(-z))

    return s

def initialize_with_zeros(dim):
    '''
    dim -- size of w vector 
    :param dim: 
    :return: 
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (bias)
    '''

    w = np.zeros((dim, 1), dtype=float, order='C')
    b = 0

    return w, b

def propagate(w, b, X, Y):
    '''
    Implement the cost function and its gradient for the propagation
    :param w: weights, a numpy array of size (num_px * num_px * 3, number of examples)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :param Y: the label vector, the correct answers of size (1, number of examples)
    :return: 
    cost --- 
    dw   --- gradient of the loss with respect to w
    db   --- gradient of b
    '''
    m = X.shape[1]
    # the shape of w is (dim, 1) , so we need the transpose of w
    A = sigmoid(np.dot(w.T, X) + b)

    # np.log(x)是以e为底x的对数
    cost = -np.sum(Y * np.log(A) + (1-Y)*np.log(1-A)) / m
    dw = np.dot(X, (A-Y).T) / m
    db = np.sum(A - Y) / m

    grads = {"dw" : dw,
             "db" : db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    '''
    The function is used to optimize w and b by running a gradient descent algorithm
    :param w: weights
    :param b: bias
    :param X: data
    :param Y: the label vector
    :param num_iterations: number of iterations of the optimization loop
    :param learning_rate: learning rate of gradient descent update rule
    :param print_cost: if True, print the loss every 100 steps
    :return: 
    params  --- dictionary containing the weights w and bias b
    grads   --- dictionary containing the gradients of the weights and bias 
    costs   --- list of all the costs computed during the optimization
    
    计算步骤：
    1、计算cost和参数的梯度，用propagate()。
    2、对w和b用梯度下降规则更新参数。
    '''
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {"w" : w,
              "b" : b}
    grads  = {"dw" : dw,
              "db" : db}
    return params, grads, costs

def predict(w, b, X):
    '''
    Predict the examples, output 0 or 1
    :param w: weights
    :param b: bias
    :param X: data of size (num_px * num_px * 3, number of examples)
    :return: 
    Y_prediction    --- a numpy array containing all predictions (0/1) for the examples
    '''
    print(w, b, X)
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid( np.dot( w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0, i] to actual predictions p[0, i]
        # 以0.5为边界
        if ( A[0][i] > 0.5):
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0

    return Y_prediction


# runs
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    '''
    Build the logistic regression model
    :param X_train: training set (num_px * num_px * 3, size of examples)
    :param Y_train: training labels
    :param X_test: test set
    :param Y_test: test labels
    :param num_iterations: 
    :param learning_rate: 
    :param print_cost: 
    :return: 
    d   --- dictionary containing information about the model
    '''

    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    d = {"costs" : costs,
         "Y_prediction_test" : Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iteration" : num_iterations}

    return d


if __name__ == "__main__":
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=True)
    print("predictions = " + str(predict(w, b, X)))
    # array A is [[ 0.99987661  0.99999386]]
    # print predictions = [[ 1.  1.]]

