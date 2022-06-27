from itertools import permutations

import names
import numpy as np
from numpy.random import choice, rand
from ConnectionGene import ConnectionGene
import NEAT


class Individual:
    def get_graph(self):
        for i in self.matrix:
            string = ""
            for j in i:
                if j is None or not j.enable:
                    string += "0" + " "
                else:
                    string += "1" + " "
            print(string)

    def __init__(self, inp=None, out=None):
        # Build from genome
        self.inp = [i for i in range(inp)]
        self.out = [i for i in range(inp, inp + out)]

        self.inp = [i for i in range(inp)]
        self.out = [i for i in range(inp, inp + out)]
        if inp + out > 5:
            self.hidden = []
        else:
            self.hidden = [i for i in range(inp + out, inp + out + 3)]
        # self.hidden = [i for i in range(inp + out, inp + out + 6)]
        self.n_neurons = inp + out + len(self.hidden)

        self.matrix = np.array([None for j in range(self.n_neurons)])
        self.genome = {}

        self.fitness = -1.
        self.adj_fitness = -1.

        self.name = names.get_last_name()

        self.sons = dict()
        for i in (self.inp + self.out):
            self.sons[i] = set()

        for i in range(self.n_neurons - 1):
            self.matrix = np.vstack([self.matrix, [None for j in range(self.n_neurons)]])

        self.log = []

    def add_connection(self, i, o, force=False):

        new = ConnectionGene(i, o)

        self.genome[new.innovation] = new
        self.matrix[i][o] = new
        NEAT.NEAT.max_len = max(NEAT.NEAT.max_len, len(self.genome.keys()))
        return new.innovation

    def add_node(self, i, o, w):
        new1 = ConnectionGene(i, self.n_neurons, 1.0)
        new2 = ConnectionGene(self.n_neurons, o, w)

        self.genome[new1.innovation] = new1
        self.genome[new2.innovation] = new2

        self.matrix = np.hstack([self.matrix, np.atleast_2d([None for j in range(self.n_neurons)]).T])
        self.matrix = np.vstack([self.matrix, [None for j in range(self.n_neurons + 1)]])

        self.matrix[i][self.n_neurons] = new1

        self.matrix[self.n_neurons][o] = new2

        self.hidden.append(self.n_neurons)

        NEAT.NEAT.max_len = max(NEAT.NEAT.max_len, len(self.genome.keys()))
        self.n_neurons += 1

    def mutate(self):
        n = len(self.genome)
        if rand() < 0.8:
            for gen in self.genome.values():
                if rand() < NEAT.NEAT.weight_mutation_rate:
                    gen.weight = gen.w + NEAT.NEAT.step * 2 * (rand() - rand())
                else:
                    gen.weight = rand() * 4 - 2

        # NEW CONNECTION
        if rand() < NEAT.NEAT.new_link_mutation_rate:
            perms = []
            perms += permutations(self.hidden, 2)
            for i in self.inp + self.hidden:
                perms += [(i, j) for j in self.out + self.hidden if i != j]
                # perms += [(i, j) for j in self.out]

            while len(perms) != 0:
                c = choice(range(0, len(perms)))
                i = perms[c][0]
                o = perms[c][1]
                if self.matrix[i][o] is None:
                    self.add_connection(i, o)
                    break
                else:
                    perms.pop(c)

        # NEW NODE
        if rand() < NEAT.NEAT.new_node_mutation_rate:
            if len(self.genome) == 0:
                return
            c = choice(list(self.genome.values()))
            while not c.enable:
                c = choice(list(self.genome.values()))
            c.enable = False
            self.add_node(c.inp, c.out, c.w)

    def force_mutate(self):
        for gen in self.genome.values():
            if rand() < NEAT.NEAT.weight_mutation_rate:
                gen.weight = gen.w + NEAT.NEAT.step * 2 * (rand() - rand()) - NEAT.NEAT.step
            else:
                gen.weight = rand() * 4 - 2

        # NEW CONNECTION
        if rand() < 1:
            perms = []
            # perms += permutations(self.hidden, 2)
            for i in self.inp + self.hidden:
                # perms += [(i, j) for j in self.out + self.hidden]
                perms += [(i, j) for j in self.out]

            while len(perms) != 0:
                c = choice(range(0, len(perms)))
                i = perms[c][0]
                o = perms[c][1]
                if self.matrix[i][o] is None:
                    self.add_connection(i, o)
                    break
                else:
                    perms.pop(c)

        # NEW NODE
        if rand() < 0.2:
            if len(self.genome) == 0:
                return
            c = choice(list(self.genome.values()))
            while not c.enable:
                c = choice(list(self.genome.values()))
            c.enable = False
            self.add_node(c.inp, c.out, c.w)

    def process(self, inputs):
        if len(inputs) != len(self.inp):
            return NotImplemented

        results = dict()
        for i, v in enumerate(inputs):
            results[i] = v

        res = []
        for i in self.out:
            res.append(NEAT.NEAT.activation_function(self.__back_rec__(i, results, [])))

        return res

    def __back_rec__(self, neuron, results, visited):
        if neuron in results.keys():
            return results[neuron]
        if neuron in visited:
            return 0

        res = 0
        for i, entry in enumerate(self.matrix[:, neuron]):
            if entry is not None:
                if entry.enable:
                    visited.append(neuron)
                    res += self.__back_rec__(i, results, visited) * entry.w
                    visited.pop()
        results[neuron] = NEAT.NEAT.inner_activation_function(res)
        return results[neuron]

    def __str__(self):
        out = self.name + " " + str(self.fitness)
        for gen in self.genome.values():
            out += chr(gen.innovation + 48) + "-"
        out += '\n'
        return out

    def calcDistance(self, ind):
        k1 = sorted(self.genome.keys())
        k2 = sorted(ind.genome.keys())
        border = min(max(k1), max(k2))
        set1 = set(k1)
        set2 = set(k2)

        disjoint1 = set1.difference(set2)
        disjoint2 = set2.difference(set1)
        differ = np.array(list(disjoint1.union(disjoint2)))

        disjoint = len(differ[differ <= border])
        excess = len(differ) - disjoint

        w_mean = 0
        intersec = list(set1.intersection(set2))
        if len(intersec) == 0:
            return np.inf

        for g in intersec:
            w_mean += abs(self.genome[g].w - ind.genome[g].w)

        w_mean /= len(intersec)
        N = 15  # if NEAT.NEAT.max_len < 15 else NEAT.NEAT.max_len
        distance = NEAT.NEAT.c1 * (excess / N) + NEAT.NEAT.c2 * (disjoint / N) + NEAT.NEAT.c3 * w_mean
        return distance
        pass

    def score(self):
        fitness = NEAT.NEAT.fitness(self)
        self.fitness = fitness
