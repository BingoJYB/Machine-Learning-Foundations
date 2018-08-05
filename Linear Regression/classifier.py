import numpy as np


def target_func(X):
    return np.sign(np.sum(X * X, axis=1) - 0.6)


def generate_data():
    x1 = np.reshape(np.random.uniform(-1, 1, 1000), (-1, 1))
    x2 = np.reshape(np.random.uniform(-1, 1, 1000), (-1, 1))
    X = np.concatenate((np.ones((1000, 1)), x1, x2), axis=1)

    Y = np.reshape(target_func(X[:, 1:3]), (-1, 1))
    Y = np.concatenate((Y[:100] * -1, Y[100:]))

    data = np.concatenate((X, Y), axis=1)
    np.random.shuffle(data)

    return data


def feature_transform(data):
    x1x2 = np.reshape(data[:, 1] * data[:, 2], (-1, 1))
    x12 = np.reshape(data[:, 1] * data[:, 1], (-1, 1))
    x22 = np.reshape(data[:, 2] * data[:, 2], (-1, 1))
    data = np.concatenate((data[:, 0:1], data[:, 1:2], data[:, 2:3],
                           x1x2, x12, x22, data[:, 3:4]), axis=1)

    return data


def calculate_Err(w, data):
    pred_val = np.sign(np.dot(data[:, :-1], w))
    Err = np.sum(pred_val != data[:, -1])

    return Err / data.shape[0]


def linear_regression(data):
    x_pseudoinv = np.linalg.pinv(data[:, :-1])
    wlin = np.dot(x_pseudoinv,  data[:, -1])

    return wlin


Ein_total = 0
wlin_with_feature_transformed_total = 0
Eout_total = 0

for i in range(1000):
    train_data = generate_data()
    wlin = linear_regression(train_data)
    Ein_total = Ein_total + calculate_Err(wlin, train_data)

    transformed_train_data = feature_transform(train_data)
    wlin_with_feature_transformed = linear_regression(transformed_train_data)
    wlin_with_feature_transformed_total = wlin_with_feature_transformed_total\
                                          + wlin_with_feature_transformed

    test_data = generate_data()
    transformed_test_data = feature_transform(test_data)
    Eout_total = Eout_total + calculate_Err(wlin_with_feature_transformed,
                                            transformed_test_data)

print(Ein_total / 1000)
print(wlin_with_feature_transformed_total / 1000)
print(Eout_total / 1000)
