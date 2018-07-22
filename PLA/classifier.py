import random
import numpy as numpy
from functools import reduce

def sign(vals):
    return numpy.asarray(list(map(lambda val: -1 if val <= 0 else 1, vals)))

def classify(iteration, factor):
    counters = []
    data = numpy.loadtxt('data.txt')
    for i in range(iteration):
        if iteration > 1:
            data = numpy.asarray(sorted(data, key = lambda k: random.random()))
        w_0 = numpy.ones((data.shape[0], 1))
        X = numpy.concatenate((w_0, data[:, :-1]), axis = 1)
        Y = data[:, -1:]
        W = numpy.zeros((data.shape[1], 1))
        isAllCorrect = False
        counter = 0
    
        while not isAllCorrect:
            isAllCorrect = True
            for row, x in enumerate(X):
                if sign(numpy.dot(x, W)) * Y[row] < 0:
                    W = W + factor * (Y[row] * x)[numpy.newaxis].T
                    isAllCorrect = False
                    counter = counter + 1
                    
        counters.append(counter)
    
    return round(reduce(lambda x, y: x + y, counters) / len(counters))

if __name__ == '__main__':
    print(classify(2000, 1))
