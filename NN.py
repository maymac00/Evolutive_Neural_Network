import numpy as np
import math


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(vector):
    vector = np.reshape(vector, (vector.shape[0]))
    e = np.exp(vector)
    return e / e.sum()


def softmax_d(signal, derivative=False):
    signal = softmax(signal)
    return signal * (1. - signal)


def Dsigmoid(output):
    return output * (1.0 - output)


def mse(v1, v2):
    return ((v1 - v2) ** 2).mean()


class Layer:
    def __init__(self, n_conections, n_neurons, activation=(lambda f: f, lambda g: g)):
        self.w = np.random.rand(n_conections, n_neurons) - np.random.rand(n_conections, n_neurons)
        self.b = np.random.rand(1, n_neurons) - np.random.rand(1, n_neurons)
        self.act_f = activation

    def forward(self, inp):
        z = np.dot(inp, self.w) + self.b
        return z, self.act_f[0](z)


class NN:
    def __init__(self, topology, activation):
        self.topology = topology
        self.layers = []

        for i in range(len(topology) - 1):
            self.layers.append(Layer(topology[i], topology[i + 1], activation[i]))

    def train(self, X, y, lr=0.1):
        out = [(None, X)]
        for l, layer in enumerate(self.layers):
            out.append(layer.forward(out[-1][1]))

        deltas = []

        z = out[-1][0]
        a = out[-1][1]
        aux = self.layers[-1].act_f[1](a)
        aux2 = (a - y)
        deltas.append(np.multiply((a - y), aux))

        for l in reversed(range(1, len(out) - 1)):
            z = out[l][0]
            a = out[l][1]

            d = np.dot(deltas[0], self.layers[l].w.T) * self.layers[l].act_f[1](a)
            deltas.insert(0, d)

        for l, layer in enumerate(self.layers):
            layer.w = layer.w - np.dot(out[l][1].T, deltas[l]) * lr
            layer.b = layer.b - np.sum(deltas[l], axis=0, keepdims=True) * lr
        return mse(out[-1][1], y)

    def predict(self, X):
        out = [(None, X)]
        for l, layer in enumerate(self.layers):
            out.append(layer.forward(out[-1][1]))
        return out[-1][1]


soft = (softmax, softmax_d)
sig = (sigmoid, Dsigmoid)
net = NN([2, 4, 1], activation=[sig, sig])
err = np.inf
it = 0
while err > 0.0001:
    it += 1
    err = net.train(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [0], [1]]))

print(err, "its: ", it)
res = net.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
print(res)
