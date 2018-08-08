import random
import numpy as numpy
from functools import reduce


def sign(vals):
    return numpy.asarray(list(map(lambda val: -1 if val <= 0 else 1, vals))).reshape((vals.shape[0] ,1))


def calculateErr(W, X, Y):
    return numpy.sum(numpy.absolute(sign(numpy.dot(X, W)) - Y) * 0.5) / X.shape[0]


def train(updates, data):
    data = numpy.asarray(sorted(data, key = lambda k: random.random()))
    w_0 = numpy.ones((data.shape[0], 1))
    X = numpy.concatenate((w_0, data[:, :-1]), axis = 1)
    Y = data[:, -1:]
    W = numpy.zeros((data.shape[1], 1))
    best_W = W
    counter = 0
    min_err = 1.0
    
    for row, x in enumerate(X):
        if sign(numpy.dot(x, W)) * Y[row] < 0:
            counter = counter + 1
            W = W + (Y[row] * x)[numpy.newaxis].T
            curr_err = calculateErr(W, X, Y)
            if curr_err < min_err:
                min_err = curr_err
                best_W = W
                
        if counter == updates:
            break
                    
    return best_W


def test(W, data):
    w_0 = numpy.ones((data.shape[0], 1))
    X = numpy.concatenate((w_0, data[:, :-1]), axis = 1)
    Y = data[:, -1:]
    return calculateErr(W, X, Y)


if __name__ == '__main__':
    training_data = numpy.loadtxt('train.txt')
    testing_data = numpy.loadtxt('test.txt')
    errs = []
    for i in range(2000):
        errs.append(test(train(100, training_data), testing_data))
        
    print(reduce(lambda x, y: x + y, errs) / len(errs))
