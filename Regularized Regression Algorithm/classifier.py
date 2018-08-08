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

# question 16
Ein_Eval_Eout_min = [np.Inf, np.Inf, np.Inf, np.Inf]
for rou in rou_list:
    wreg = get_wreg(train_data[:120, :], np.power(np.asarray([10], dtype='d'), rou))
    Ein = calculate_Err(wreg, train_data[:120, :])
    Eval = calculate_Err(wreg, train_data[120:, :])
    Eout = calculate_Err(wreg, test_data)

    if Ein <= Ein_Eval_Eout_min[1]:
        Ein_Eval_Eout_min[0] = rou
        Ein_Eval_Eout_min[1] = Ein
        Ein_Eval_Eout_min[2] = Eval
        Ein_Eval_Eout_min[3] = Eout

print(Ein_Eval_Eout_min)

# question 17
Ein_Eval_Eout_min = [np.Inf, np.Inf, np.Inf, np.Inf]
for rou in rou_list:
    wreg = get_wreg(train_data[:120, :], np.power(np.asarray([10], dtype='d'), rou))
    Ein = calculate_Err(wreg, train_data[:120, :])
    Eval = calculate_Err(wreg, train_data[120:, :])
    Eout = calculate_Err(wreg, test_data)

    if Eval <= Ein_Eval_Eout_min[2]:
        Ein_Eval_Eout_min[0] = rou
        Ein_Eval_Eout_min[1] = Ein
        Ein_Eval_Eout_min[2] = Eval
        Ein_Eval_Eout_min[3] = Eout

print(Ein_Eval_Eout_min)

# question 18
wreg = get_wreg(train_data, np.power(np.asarray([10], dtype='d'), Ein_Eval_Eout_min[0]))
Ein = calculate_Err(wreg, train_data)
Eval = calculate_Err(wreg, train_data)
Eout = calculate_Err(wreg, test_data)

print((Ein, Eout))

# question 19
rou_optimal = 0
Ecv_total_min = np.inf
for rou in rou_list:
    Ecv_total = 0

    for i in range(5):
        masked_train_data = np.ma.array(train_data, mask=False)
        masked_train_data.mask[i*40:(i+1)*40] = True

        wreg = get_wreg(np.asmatrix(masked_train_data), np.power(np.asarray([10], dtype='d'), rou))
        Ecv = calculate_Err(wreg, train_data[i*40:(i+1)*40, :])
        Ecv_total = Ecv_total + Ecv

    Ecv_total = Ecv_total / 5

    if Ecv_total <= Ecv_total_min:
        rou_optimal = rou
        Ecv_total_min = Ecv_total

print((rou_optimal, Ecv_total_min))

# question 20
wreg = get_wreg(train_data, np.power(np.asarray([10], dtype='d'), rou_optimal))
Ein = calculate_Err(wreg, train_data)
Eout = calculate_Err(wreg, test_data)

print((Ein, Eout))


Ecv_total = 0

mask = [0] * 800
mask[160:320] = [1] * 160
masked_train_data = np.ma.array(train_data, mask=mask)

wreg = get_wreg(train_data[40:, :], np.power(np.asarray([10], dtype='d'), -8.0))
Ecv = calculate_Err(wreg, train_data[0:40, :])
Ecv_total = Ecv_total + Ecv

print(masked_train_data)
wreg = get_wreg(np.asmatrix(masked_train_data), np.power(np.asarray([10], dtype='d'), -8.0))
Ecv = calculate_Err(wreg, train_data[40:80, :])
Ecv_total = Ecv_total + Ecv

wreg = get_wreg(np.asmatrix(np.concatenate((np.asarray(train_data[:80, :]), np.asarray(train_data[120:, :])))), np.power(np.asarray([10], dtype='d'), -8.0))
Ecv = calculate_Err(wreg, train_data[80:120, :])
Ecv_total = Ecv_total + Ecv

wreg = get_wreg(np.asmatrix(np.concatenate((np.asarray(train_data[:120, :]), np.asarray(train_data[160:, :])))), np.power(np.asarray([10], dtype='d'), -8.0))
Ecv = calculate_Err(wreg, train_data[120:160, :])
Ecv_total = Ecv_total + Ecv

wreg = get_wreg(train_data[:160, :], np.power(np.asarray([10], dtype='d'), -8.0))
Ecv = calculate_Err(wreg, train_data[160:200, :])
Ecv_total = Ecv_total + Ecv

print(Ecv_total / 5)
