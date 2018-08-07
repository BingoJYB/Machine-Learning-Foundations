import numpy as np


def readData(filename):
    data = np.loadtxt(filename)
    data = np.insert(data, 0, 1, axis=1)
    return np.asmatrix(data)


def get_wreg(data, rou):
    X = data[:, :-1]
    y = data[:, -1:]

    wreg = (X.T * X + rou * np.identity(X.shape[1])).getI() * X.T * y

    return wreg


def calculate_Err(w, data):
    pred_val = np.sign(data[:, :-1].dot(w))
    Err = np.sum(pred_val != data[:, -1])

    return Err / data.shape[0]


train_data = readData('train.txt')
test_data = readData('test.txt')

# question 13
rou = 10
wreg = get_wreg(train_data, rou)
Ein = calculate_Err(wreg, train_data)
Eout = calculate_Err(wreg, test_data)
print(Ein)
print(Eout)

# question 14
rou_list = np.asarray(range(-10, 3), dtype='d')
Ein_Eout_min = [np.Inf, np.Inf, np.Inf]
for rou in rou_list:
    wreg = get_wreg(train_data, np.power(np.asarray([10], dtype='d'), rou))
    Ein = calculate_Err(wreg, train_data)
    Eout = calculate_Err(wreg, test_data)

    if Ein <= Ein_Eout_min[1]:
        Ein_Eout_min[0] = rou
        Ein_Eout_min[1] = Ein
        Ein_Eout_min[2] = Eout

print(Ein_Eout_min)

# question 15
Ein_Eout_min = [np.Inf, np.Inf, np.Inf]
for rou in rou_list:
    wreg = get_wreg(train_data, np.power(np.asarray([10], dtype='d'), rou))
    Ein = calculate_Err(wreg, train_data)
    Eout = calculate_Err(wreg, test_data)

    if Eout <= Ein_Eout_min[2]:
        Ein_Eout_min[0] = rou
        Ein_Eout_min[1] = Ein
        Ein_Eout_min[2] = Eout

print(Ein_Eout_min)
