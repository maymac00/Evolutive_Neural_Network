import json

import numpy as np

from ConnectionGene import ConnectionGene
from Individual import Individual

"""

    This class is used to build a new individual from a genome either randomly, from a file or from a new genome.
"""


class IndividualFactory:
    def_input = 3
    def_output = 1

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

    @staticmethod
    def buildRandomLike(param):
        n = len(param.genome.keys())
        ind = Individual(len(param.inp), len(param.out))
        while len(ind.genome.keys()) < n:
            ind.force_mutate()
        return ind
        pass

    # read individual from file
    @staticmethod
    def buildFromFile(filename):
        json_raw = open(filename, 'r').read()
        json_data = json.loads(json_raw)
        gens = []
        json_genome = json_data['genome']
        for gen in json_genome:
            gens.append(ConnectionGene(gen['inp'], gen['out'], gen['w'], gen['enable'], gen['innovation']))

        ind = IndividualFactory.buildIndividual(IndividualFactory.def_input, IndividualFactory.def_output, gens)
        ind.fitness = json_data['fitness']
        ind.name = json_data['name']
        return ind
