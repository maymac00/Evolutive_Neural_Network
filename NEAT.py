import numpy as np
from numpy.random import rand

import copy

"""

    This class contains the NEAT constants. For example, the distnace constants, the step for weight mutation, 
    mutation rates, target species, etc.
    
    The class also contains some core funtions for the NEAT algorithm. Such as the crossover function, and all the
    activation functions. It also hosts as a variable the current fitness function to be evaluated.
"""


class NEAT:
    seed = -1
    adaptation = 0.3
    dropoff = 15
    blood_rate = 3
    max_rwd = 500
    new_node_mutation_rate = 0.05
    new_link_mutation_rate = 0.08
    weight_mutation_rate = 0.9
    n_innovations = 0
    innovations = dict()
    max_len = 0
    step = 1
    def_step = step
    species_pool_size = 12
    reps = 1
    opt = "max"

    game = "CartPole-v0"

    distance_thld = 10.0
    def_distance_thld = distance_thld
    c1 = 2
    c2 = 2
    c3 = 1

    @staticmethod
    def normalize01(data, min=None, max=None):
        if min is not None:
            return (data - min) / (max - min)
        if data.max() - data.min() == 0:
            return np.ones_like(data)
        return (data - data.min()) / (data.max() - data.min())

    # normalize data to [-1,1]
    @staticmethod
    def normalize_1_1(data, min=None, max=None):
        return NEAT.normalize01(data, min=min, max=max) * 2 - 1

    @staticmethod
    def norm_variance(data, min, max):
        if np.var([min, max]) == 0:
            return np.zeros_like(data)

        return (data - min) / np.var([min, max])

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
    def relu(x):
        return max(0, x)

    @staticmethod
    def sigmoid_mod(x):
        z = np.exp(-x)
        sig = 4 / (1 + z)
        return sig - 2

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def hiperbolic_tangent(data):
        return np.tanh(data)

    activation_function = sigmoid
    inner_activation_function = sigmoid

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
        import IndividualFactory
        ind = IndividualFactory.IndividualFactory.buildIndividual(len(ind1.inp), len(ind1.out), new_genome)
        return ind
        pass

    pass

    fitness = None
