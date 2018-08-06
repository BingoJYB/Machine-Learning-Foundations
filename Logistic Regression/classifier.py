import numpy as np


def readData(filename):
    data = np.loadtxt(filename)
    return data


def init_w(data):
    return np.zeros(data.shape[1])


def h(s):
    return 1 / (1 + np.exp(-s))


def gradient_Ein(w, data):
    X = data[:, :-1]
    X = np.insert(X, 0, 1, axis=1)
    Y = data[:, -1:]

    gradient_Ein = np.sum(h(-Y * np.dot(X, w.reshape((-1, 1))))
                          * (-Y * X), axis=0) / data.shape[0]

    return gradient_Ein


def stochastic_gradient_Ein(w, data, i):
    X = data[:, :-1]
    X = np.insert(X, 0, 1, axis=1)
    Y = data[:, -1:]
    i = i % data.shape[0]

    gradient_Ein = h(-Y[i] * np.dot(X[i], w.reshape((-1, 1)))) * (-Y[i] * X[i])

    return gradient_Ein


def calculate_Eout(w, data):
    X = data[:, :-1]
    X = np.insert(X, 0, 1, axis=1)
    Y = data[:, -1:]

    pred_val = np.sign(np.dot(X, w.reshape((-1, 1))))
    Err = np.sum(pred_val != Y)

    return Err / data.shape[0]


train_data = readData("train.txt")
test_data = readData("test.txt")
w = init_w(train_data)
stochastic_w = init_w(train_data)

for i in range(2000):
    w = w - 0.001 * gradient_Ein(w, train_data)
    stochastic_w = stochastic_w - 0.001 *\
                   stochastic_gradient_Ein(stochastic_w, train_data, i)

Eout = calculate_Eout(w, test_data)
stochastic_Eout = calculate_Eout(stochastic_w, test_data)

print(Eout)
print(stochastic_Eout)