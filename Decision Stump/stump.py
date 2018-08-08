import numpy as np


def signWithNoise(X, p):
    isFlip = np.random.choice([1, -1], 20, p=[1 - p, p])
    Y = X * isFlip
    return Y


def readData(filename):
    X = [[] for _ in range(9)]
    Y = []

    with open(filename) as f:
        content = f.readlines()

    for str in content:
        sample = str.split()
        Y.append(sample[-1])
        for idx, val in enumerate(sample[:-1]):
            X[idx].append(val)

    return X, Y


def generateData(num):
    X = np.random.uniform(-1, 1, num)
    temp_X = np.copy(X)
    temp_X[temp_X >= 0] = 1
    temp_X[temp_X < 0] = -1
    Y = signWithNoise(temp_X, 0.2)
    return X, Y


def calculateOneDimEin(X, Y):
    min_Ein = np.inf
    best_s = 0
    best_theta = 0

    for s in [1, -1]:
        for theta in np.nditer(X):
            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = 1 * s
            temp_X[temp_X < theta] = -1 * s
            Ein1 = np.sum(temp_X != Y) / X.shape[0]

            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = -1 * s
            temp_X[temp_X < theta] = 1 * s
            Ein2 = np.sum(temp_X != Y) / X.shape[0]

            Ein = min(Ein1, Ein2)
            if Ein < min_Ein:
                min_Ein = Ein
                best_s = s
                best_theta = abs(theta)

    return min_Ein, best_s, best_theta


def calculateMultiDimEin(X, Y):
    Eins = []
    ss = []
    thetas = []
    min_Ein = np.inf
    best_s = 0
    best_theta = 0
    best_dim = 0
    Y = np.asarray(Y, dtype=float)

    for x in X:
        x = np.asarray(x, dtype=float)
        Ein, s, theta = calculateOneDimEin(x, Y)
        Eins.append(Ein)
        ss.append(s)
        thetas.append(theta)

    for idx, val in enumerate(Eins):
        if val < min_Ein:
            min_Ein = val
            best_s = ss[idx]
            best_theta = thetas[idx]
            best_dim = idx

    return min_Ein, best_s, best_theta, best_dim


def _17_18(times):
    all_Ein = 0
    all_Eout = 0

    for i in range(times):
        X, Y = generateData(20)
        Ein, s, theta = calculateOneDimEin(X, Y)
        all_Ein = all_Ein + Ein
        all_Eout = all_Eout + 0.5 + 0.3 * s * (theta - 1)

    print(all_Ein / times)
    print(all_Eout / times)


def _19_20():
    train_X, train_Y = readData('train.txt')
    test_X, test_Y = readData('test.txt')
    Ein, s, theta, dim = calculateMultiDimEin(train_X, train_Y)
    test_X_dim = np.asarray(test_X[dim], dtype=float)
    Y1 = []
    Y2 = []
    for x in np.nditer(test_X_dim):
        if x >= theta:
            Y1.append(s)
            Y2.append(-s)
        else:
            Y1.append(-s)
            Y2.append(s)
    Y1 = np.asarray(Y1, dtype=float)
    Y2 = np.asarray(Y2, dtype=float)
    test_Y = np.asarray(test_Y, dtype=float)
    Eout = min(np.sum(test_Y != Y1) / Y1.shape[0], np.sum(test_Y != Y2) / Y2.shape[0])
    print(Ein)
    print(Eout)

    
_17_18(5000)
_19_20()
