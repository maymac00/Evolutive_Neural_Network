from NN import NN
import numpy as np


class Generation:
    def __init__(self, n_individuals, base_net):
        self.n = n_individuals
        self.population = [NN(base_net.topology, activation=base_net.activation) for i in range(n_individuals)]
