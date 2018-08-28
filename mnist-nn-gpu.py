#!/usr/bin/python
#
# MNIST digit recognizer in GPU mode
#

import numpy as np
import math
import pandas as pd
import scipy.io as sio
import theano
import theano.tensor as T
import time

# Structure of the 3-layer neural network.
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

# Matrix product function.  Default is to use CPU mode.
Matrix_dot = np.dot


def load_training_data(training_file='mnistdata.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).

    inputs: numpy array with size (5000, 400).
    labels: numpy array with size (5000, 1).

    The training data is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4data1.mat).
    '''
    training_data = sio.loadmat(training_file)
    return (training_data['X'], training_data['y'])


def load_weights(weight_file='mnistweights.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).

    The weights file is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4weights.mat).
    '''
    weights = sio.loadmat(weight_file)
    return weights


def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


def sigmoid(z):
    return 1.0 / (1 + pow(math.e, -z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def gpu_matrix_dot():
    time_start = time.time()
    x = T.matrix('x')
    y = T.matrix('y')
    z = T.dot(x, y)
    f = theano.function([x, y], z, allow_input_downcast=True)
    time_end = time.time()
    print('theano expression creation costs {} secs'.format(time_end - time_start))
    return f


def cost_function(theta1, theta2, input_layer_size, hidden_layer_size, output_layer_size, inputs, labels, regular=0):
    '''
    Note: theta1, theta2, inputs, labels are numpy arrays:

        theta1: (25, 401)
        theta2: (10, 26)
        inputs: (5000, 400)
        labels: (5000, 1)
    '''
    # construct neural network
    input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401

    time_start = time.time()
    hidden_layer = Matrix_dot(input_layer, theta1.T)
    hidden_layer = sigmoid(hidden_layer)
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)  # add bias, 5000x26
    time_end = time.time()
    print('\tconstruction: hidden layer dot costs {} secs'.format(time_end - time_start))

    time_start = time.time()
    output_layer = Matrix_dot(hidden_layer, theta2.T)  # 5000x10
    output_layer = sigmoid(output_layer)
    time_end = time.time()
    print('\tconstruction: output layer dot costs {} secs'.format(time_end - time_start))

    # forward propagation: calculate cost
    time_start = time.time()
    cost = 0.0
    for training_index in xrange(len(inputs)):
        outputs = [0] * output_layer_size
        outputs[labels[training_index]-1] = 1

        for k in xrange(output_layer_size):
            cost += -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
    cost /= len(inputs)
    time_end = time.time()
    print('\tforward prop: costs {} secs'.format(time_end - time_start))

    # back propagation: calculate gradiants
    time_start = time.time()
    theta1_grad = np.zeros_like(theta1)  # 25x401
    theta2_grad = np.zeros_like(theta2)  # 10x26
    for index in xrange(len(inputs)):
        # transform label y[i] from a number to a vector.
        outputs = np.zeros((1, output_layer_size))  # (1,10)
        outputs[0][labels[index]-1] = 1

        # calculate delta3
        delta3 = (output_layer[index] - outputs).T  # (10,1)

        # calculate delta2
        z2 = Matrix_dot(theta1, input_layer[index:index+1].T)  # (25,401) x (401,1)
        z2 = np.insert(z2, 0, 1, axis=0)  # add bias, (26,1)
        delta2 = np.multiply(
            Matrix_dot(theta2.T, delta3),  # (26,10) x (10,1)
            sigmoid_gradient(z2)  # (26,1)
        )
        delta2 = delta2[1:]  # (25,1)

        # calculate gradients of theta1 and theta2
        # (25,401) = (25,1) x (1,401)
        theta1_grad += Matrix_dot(delta2, input_layer[index:index+1])
        # (10,26) = (10,1) x (1,26)
        theta2_grad += Matrix_dot(delta3, hidden_layer[index:index+1])
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)
    time_end = time.time()
    print('\tback prop: costs {} secs'.format(time_end - time_start))

    return cost, (theta1_grad, theta2_grad)


def gradient_descent(inputs, labels, learningrate=0.8, iteration=50):
    '''
    @return cost and trained model (weights).
    '''
    rand_theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
    rand_theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    theta1 = rand_theta1
    theta2 = rand_theta2
    cost = 0.0
    for i in xrange(iteration):
        time_start = time.time()
        cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            Input_layer_size, Hidden_layer_size, Output_layer_size,
            inputs, labels, regular=0)
        theta1 -= learningrate * theta1_grad
        theta2 -= learningrate * theta2_grad
        time_end = time.time()
        print('Iteration {0} (learning rate {2}, iteration {3}), cost: {1}, time: {4}'.format(i+1, cost, learningrate, iteration, time_end - time_start))
    return cost, (theta1, theta2)


def train(inputs, labels, learningrate=0.8, iteration=50):
    cost, model = gradient_descent(inputs, labels, learningrate, iteration)
    return model


def predict(model, inputs):
    theta1, theta2 = model
    a1 = np.insert(inputs, 0, 1, axis=1)  # add bias, (5000,401)
    a2 = np.dot(a1, theta1.T)  # (5000,401) x (401,25)
    a2 = sigmoid(a2)
    a2 = np.insert(a2, 0, 1, axis=1)  # add bias, (5000,26)
    a3 = np.dot(a2, theta2.T)  # (5000,26) x (26,10)
    a3 = sigmoid(a3)  # (5000,10)
    return [i.argmax()+1 for i in a3]


if __name__ == '__main__':
    gpu_mode = True
    #gpu_mode = False
    if gpu_mode is True:
        print('GPU mode')
        Matrix_dot = gpu_matrix_dot()
    else:
        print('CPU mode')
        Matrix_dot = np.dot

    # Note: There are 10 units which present the digits [1-9, 0]
    # (in order) in the output layer.
    inputs, labels = load_training_data()

    # (optional) load pre-trained model for debugging neural network construction
    weights = load_weights()
    theta1 = weights['Theta1']  # size: 25 entries, each has 401 numbers
    theta2 = weights['Theta2']  # size: 10 entries, each has  26 numbers

    theta1 = np.array(theta1)
    theta2 = np.array(theta2)
 
    cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, Input_layer_size, Hidden_layer_size, Output_layer_size, inputs, labels, regular=0)
    print('cost:', cost)

    # train the model from scratch and predict based on it
    # learning rate 0.10, iteration  60: 36% (cost: 3.217)
    # learning rate 1.75, iteration  50: 77%
    # learning rate 1.90, iteration  50: 75%
    # learning rate 2.00, iteration  50: 82%
    # learning rate 2.00, iteration 100: 87%
    # learning rate 2.00, iteration 200: 93% (cost: 0.572)
    # learning rate 2.00, iteration 300: 94% (cost: 0.485)
    # learning rate 2.05, iteration  50: 79%
    # learning rate 2.20, iteration  50: 64%
    model = train(inputs, labels, learningrate=0.1, iteration=60)
    #model = (theta1, theta2)
    outputs = predict(model, inputs)

    correct_prediction = 0
    for i, predict in enumerate(outputs):
        if predict == labels[i][0]:
            correct_prediction += 1
    precision = float(correct_prediction) / len(labels)
    print('precision: {}'.format(precision))
