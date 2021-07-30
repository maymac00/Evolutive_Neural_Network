from NN import NN
import numpy as np
from numpy.random import choice, rand


class ConnectionGene:
    def __init__(self, weight=rand(), enable=True):
        self.w = weight
        self.enable = enable
        self.innovation = NEAT.n_innovations
        NEAT.n_innovations += 1


class ConnectionGenes:

    def __init__(self, inp, out):
        self.inp = [i for i in range(inp)]
        self.out = [i for i in range(inp, inp + out)]
        self.hidden = []

        self.matrix = []
        self.genome = []

        for i in range(inp + out):
            self.matrix.append([])
            for j in range(inp + out):
                self.matrix[i].append(None)

        for a in self.inp:
            for b in self.out:
                self.matrix[a][b] = ConnectionGene()
                self.matrix[b][a] = self.matrix[a][b]
                self.genome.append(self.matrix[a][b])
        pass

    def new_connection(self):
        if len(self.hidden) == 0:
            return

        i = choice(self.inp + self.hidden)
        o = choice(self.out + self.hidden)

        if self.matrix[i][o] is None:
            self.matrix[i][o] = ConnectionGene()
            self.matrix[o][i] = self.matrix[i][o]


class NEAT:
    n_innovations = 1

    def __init__(self, inp, out):
        self.connections = ConnectionGenes(inp, out)


class Generation:
    def __init__(self, n_individuals, inp, out):
        self.n = n_individuals
        self.population = [NEAT(inp, out) for i in range(n_individuals)]
