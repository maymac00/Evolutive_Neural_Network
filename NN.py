import numpy as np
import math
from activation_functions import sigmoid, relu, softmax
import random


def mse(v1, v2):
    return ((v1 - v2) ** 2).mean()


class Layer:
    def __init__(self, n_conections, n_neurons, activation=(lambda f: f, lambda g: g)):
        self.n_conections = n_conections
        self.n_neurons = n_neurons
        self.w = np.random.rand(n_conections, n_neurons) - np.random.rand(n_conections, n_neurons)
        self.b = np.random.rand(1, n_neurons) - np.random.rand(1, n_neurons)
        self.act_f = activation

    def forward(self, inp):
        z = np.dot(inp, self.w) + self.b
        return z, self.act_f[0](z)

    def reset(self):
        self.w = np.random.rand(self.n_conections, self.n_neurons) - np.random.rand(self.n_conections, self.n_neurons)
        self.b = np.random.rand(1, self.n_neurons) - np.random.rand(1, self.n_neurons)


class NN:
    def __init__(self, topology, activation):
        self.topology = topology
        self.layers = []
        self.activation = activation
        for i in range(len(topology) - 1):
            act = sigmoid
            if activation[i] == 'relu': act = relu
            if activation[i] == 'softmax': act = softmax
            self.layers.append(Layer(topology[i], topology[i + 1], act))

    def fit(self, X, y, lr=0.1):
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

    def train(self, X, y, lr=0.1, max_it=50000, err_lim=0.0001):
        err = np.inf
        last_err = np.nan
        it = 0
        while err_lim < err != last_err and it < max_it:
            last_err = err
            it += 1
            err = self.fit(X, y, lr=lr)
        return err, it

    def addNeuron(self):
        n_layers = len(self.layers)
        if n_layers - 2 == 0:
            l = 0
        else:
            l = random.choice(range(0, n_layers - 2))

        s = self.layers[l].w.shape[0]
        add = np.random.rand(s, 1) - np.random.rand(s, 1)
        self.layers[l].w = np.append(self.layers[l].w, add, axis=1)
        self.layers[l].b = np.append(self.layers[l].b, np.random.rand(1, 1) - np.random.rand(1, 1), axis=1)

        s1 = self.layers[l + 1].w.shape[1]
        add = np.random.rand(s1, 1) - np.random.rand(s1, 1)
        self.layers[l + 1].w = np.append(self.layers[l + 1].w, add)
        s2 = int(self.layers[l + 1].w.shape[0] / s1)
        self.layers[l + 1].w = np.reshape(self.layers[l + 1].w, (s2, s1))

        self.topology[l] += 1

    def predict(self, X):
        out = [(None, X)]
        for l, layer in enumerate(self.layers):
            out.append(layer.forward(out[-1][1]))
        return out[-1][1]

    def reset(self):
        for l in self.layers:
            l.reset()
