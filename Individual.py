import numpy as np
from numpy.random import choice, rand
import copy
import gym
import time


class NEAT:
    new_node_mutation_rate = 0.6
    new_link_mutation_rate = 0.5
    weight_mutation_rate = 0.7
    n_innovations = 0
    innovations = dict()
    max_len = 0
    step = 0.05

    @staticmethod
    def normalize(data):
        return (data - data.min()) / (data.max() - data.min())

    @staticmethod
    def norm_variance(data, min, max):
        return (data - min) / np.var([min,  max])

    distance_thld = 4.0
    uslessness_thld = 10
    c1 = 0.4
    c2 = 0.4
    c3 = 0.2

    x = None
    y = None

    @staticmethod
    def check_innovation(gen):
        if gen in NEAT.innovations.keys():
            gen.innovation = NEAT.innovations[gen]
        else:
            NEAT.innovations[gen] = NEAT.n_innovations
            gen.innovation = NEAT.n_innovations
            NEAT.n_innovations += 1

    @staticmethod
    def sigmoid(x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    @staticmethod
    def sigmoid_mod(x):
        z = np.exp(-x)
        sig = 4 / (1 + z)
        return sig - 2

    activation_function = sigmoid

    @staticmethod
    def regular(x):
        return x

    @staticmethod
    def crossover(ind1, ind2):
        if ind1.fitness > ind2.fitness:
            temp = ind1
            ind1 = ind2
            ind2 = temp

        genome1 = ind1.genome.keys()
        genome2 = ind2.genome.keys()

        new_genome = []

        for i, gen in enumerate(genome1):
            if gen in ind2.genome.keys() and rand() < 0.5 and ind2.genome[gen].enable:
                new_genome.append(copy.deepcopy(ind2.genome[gen]))
            else:
                new_genome.append(copy.deepcopy(ind1.genome[gen]))

        return IndividualFactory.buildIndividual(len(ind1.inp), len(ind1.out), new_genome)
        pass

    pass

    @staticmethod
    def fitness(ind, render=False):
        rew = 0.
        env = gym.make("CartPole-v1")
        # env = gym.make("Pendulum-v1") # NEAT.activation_function = NEAT.sigmoid_mod

        action = 0
        env.seed(1)
        env.reset()

        for _ in range(1000):
            if render:
                time.sleep(0.05)
                env.render()
            # action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)

            res = ind.process(observation)
            # Cartpole-v1 #
            action = res.index(max(res))
            # action = np.array(res)
            rew += reward
            if done:
                break
        env.close()
        if render:
            time.sleep(1)
            print(rew)
        return rew


class ConnectionGene:
    def __init__(self, i, o, weight=rand(), enable=True, ):
        self.inp = i
        self.out = o
        self.w = weight
        self.enable = enable
        self.innovation = -1
        NEAT.check_innovation(self)

    def __eq__(self, other):
        if not isinstance(other, ConnectionGene):
            NotImplemented
        return self.inp == other.inp and self.out == other.out

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.inp, self.out))


class IndividualFactory:
    @staticmethod
    def createIndividual(inp, out):
        return Individual(inp, out)

    @staticmethod
    def buildIndividual(inp, out, genome):
        ind = Individual(inp, out)
        for gen in genome:
            if gen.out not in (ind.out + ind.hidden):
                ind.matrix = np.hstack([ind.matrix, np.atleast_2d([None for j in range(ind.n_neurons)]).T])
                ind.matrix = np.vstack([ind.matrix, [None for j in range(ind.n_neurons + 1)]])
                ind.hidden.append(gen.out)
                ind.n_neurons += 1
            n = ind.add_connection(gen.inp, gen.out, force=True)
            ind.genome[n].w = gen.w
            ind.genome[n].enable = gen.enable
            ind.genome[n].innovation = gen.innovation
        return ind


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
            self.hidden = [i for i in range(inp + out, inp + out + 6)]
        # self.hidden = [i for i in range(inp + out, inp + out + 6)]
        self.n_neurons = inp + out + len(self.hidden)

        self.matrix = np.array([None for j in range(self.n_neurons)])
        self.genome = {}

        self.fitness = -1.
        self.adj_fitness = -1.

        self.sons = dict()
        for i in (self.inp + self.out):
            self.sons[i] = set()

        for i in range(self.n_neurons - 1):
            self.matrix = np.vstack([self.matrix, [None for j in range(self.n_neurons)]])

    def add_connection(self, i, o, force=False):

        new = ConnectionGene(i, o)

        old_genome = copy.copy(self.genome)
        old_matrix = copy.copy(self.matrix)
        self.genome[new.innovation] = new
        self.matrix[i][o] = new
        if NEAT.fitness(self) > self.fitness * 0.9 or force:

            NEAT.max_len = max(NEAT.max_len, len(self.genome.keys()))
            return new.innovation
        else:
            self.genome = old_genome
            self.matrix = old_matrix
            return -1

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

        NEAT.max_len = max(NEAT.max_len, len(self.genome.keys()))
        self.n_neurons += 1

    def mutate(self):
        if rand() < NEAT.weight_mutation_rate:
            for gen in self.genome.values():
                gen.weight = gen.w + NEAT.step * (rand() - rand())

        # NEW CONNECTION
        if rand() < NEAT.new_link_mutation_rate:
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
        if rand() < NEAT.new_node_mutation_rate:
            if len(self.genome) == 0:
                return
            c = choice(list(self.genome.values()))
            while not c.enable:
                c = choice(list(self.genome.values()))
            c.enable = False
            self.add_node(c.inp, c.out, c.w)

    def force_mutate(self):
        step = 0.1
        r = rand()
        if r < 1:
            for gen in self.genome.values():
                gen.weight = gen.w + step * (rand() - rand())
        else:
            for gen in self.genome.values():
                gen.w = rand() * 2 - 1

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
        if rand() < 1:
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
            res.append(NEAT.activation_function(self.__back_rec__(i, results)))

        return res

    def __back_rec__(self, neuron, results):
        if neuron in results.keys():
            return results[neuron]
        res = 0
        for i, entry in enumerate(self.matrix[:, neuron]):
            if entry is not None:
                if entry.enable:
                    res += self.__back_rec__(i, results) * entry.w
        results[neuron] = NEAT.sigmoid(res)
        return results[neuron]

    def __str__(self):
        out = str(self.fitness)
        for gen in self.genome.values():
            out += chr(gen.innovation + 48) + "-"
        out += '\n'
        return out

    def check_cycles(self):
        mask = np.reshape(copy.copy(self.matrix), self.n_neurons * self.n_neurons)
        mask = list(map(lambda t: t != None and t.enable, mask))
        mat = np.zeros_like(mask)
        mat[mask] = 1
        mat = np.reshape(mat, (self.n_neurons, self.n_neurons)).astype("int8")
        mat = np.logical_or(mat, mat.T)

        deg_mat = np.zeros_like(mat, dtype="int8")
        for i in range(deg_mat.shape[0]):
            deg_mat[i, i] = np.sum(mat[i])

        laplace = np.subtract(deg_mat, mat)
        b = np.trace(laplace) < 2 * (np.linalg.matrix_rank(laplace) + 1)
        return b
        pass

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
        N = 1 if NEAT.max_len < 15 else NEAT.max_len
        distance = NEAT.c1 * (excess / N) + NEAT.c2 * (disjoint / N) + NEAT.c3 * (w_mean)
        return distance
        pass

    def score(self):
        fitness = NEAT.fitness(self)
        self.fitness = fitness
