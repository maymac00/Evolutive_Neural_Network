from NN import NN
import numpy as np
from numpy.random import choice, rand
from itertools import permutations


class ConnectionGene:
    def __init__(self, i, o, weight=rand(), enable=True):
        self.inp = i
        self.out = o
        self.w = weight
        self.enable = enable
        self.innovation = NEAT.n_innovations
        NEAT.n_innovations += 1


class Individual:

    def __init__(self, inp, out):
        self.inp = [i for i in range(inp)]
        self.out = [i for i in range(inp, inp + out)]
        self.hidden = []

        self.n_neurons = inp + out
        self.cont = 0
        self.matrix = np.array([None for j in range(self.n_neurons)])
        self.genome = []

        for i in range(self.n_neurons - 1):
            self.matrix = np.vstack([self.matrix, [None for j in range(self.n_neurons)]])
        pass

    def add_connection(self):
        perms = []
        perms += permutations(self.hidden, 2)
        for i in self.inp:
            perms += [(i, j) for j in self.out + self.hidden]

        while len(perms) != 0:
            c = choice(range(0, len(perms)))
            i = perms[c][0]
            o = perms[c][1]
            if self.matrix[i][o] is None:
                new = ConnectionGene(i, o)
                self.genome.append(new)
                self.matrix[i][o] = new
                self.matrix[o][i] = self.matrix[i][o]
                break
            else:
                perms.pop(c)

    def add_node(self):
        c = choice(self.genome)
        while not c.enable:
            c = choice(range(len(self.genome)))
        c.enabled = False

        new1 = ConnectionGene(c.inp, self.n_neurons, 1.0)
        new2 = ConnectionGene(self.n_neurons, c.out, c.w)

        self.matrix = np.hstack([self.matrix, np.atleast_2d([None for j in range(self.n_neurons)]).T])
        self.matrix = np.vstack([self.matrix, [None for j in range(self.n_neurons + 1)]])

        self.matrix[c.inp][self.n_neurons] = new1
        self.matrix[self.n_neurons][c.inp] = new1

        self.matrix[self.n_neurons][c.out] = new2
        self.matrix[c.out][self.n_neurons] = new2

        self.n_neurons += 1


class Generation:
    def __init__(self, n_individuals, inp, out):
        self.n = n_individuals
        self.population = [Individual(inp, out) for i in range(n_individuals)]


class NEAT:
    n_innovations = 1
