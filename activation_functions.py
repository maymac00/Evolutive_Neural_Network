import numpy as np


def f_relu(x):
    return np.maximum(np.zeros(x.shape), x)


def d_relu(x):
    res = np.zeros(x.shape)
    ind = np.where(x >= 0)
    res[ind[0]] = 1
    return res


def f_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def f_softmax(vector):
    vector = np.reshape(vector, (vector.shape[0], vector.shape[1]))
    e = np.exp(vector)
    return e / e.sum()


def d_softmax(signal):
    signal = f_softmax(signal)
    return signal * (1. - signal)


def d_sigmoid(output):
    return output * (1.0 - output)


relu = (f_relu, d_relu)
softmax = (f_softmax, d_softmax)
sigmoid = (f_sigmoid, d_sigmoid)
